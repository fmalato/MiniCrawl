import datetime
import json
import os
import socket
import time
import random

import matplotlib.pyplot as plt
import numpy as np

import mss
import threading

from minicrawl.dungeon_master import DungeonMaster
from tqdm import tqdm
from PIL import Image
import ffmpeg

import logging
logger = logging.getLogger()
logger.setLevel("INFO")

HOSTNAME = "127.0.0.1"
PORT = 5555

ORIENTATIONS = {
    1: {
        "south": 0,
        "west": 1,
        "north": 2,
        "east": 3
    },
    2: {
        "south,west": 0,
        "south,north": 0,
        "west,north": 1,
        "west,east": 1,
        "north,east": 2,
        "east,south": 3
    },
    3: {
        "east,south,west": 0,
        "south,west,north": 1,
        "west,north,east": 2,
        "north,east,south": 3
    }
}


def generate_result_folder_name():
    return ''.join(random.choice('0123456789ABCDEFGHIJLKMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz') for i in range(24))


class GymFlaskServer:

    def __init__(self, record=False):
        self.socket = None
        self.disconnect_requested = False
        self.pid = os.getpid()
        self.window_pos_x = None
        self.window_pos_y = None
        self.window_width = None
        self.window_height = None
        self.monitor = None
        self.image = None
        self.dungeon_master = DungeonMaster(starting_grid_size=3, max_grid_size=None, increment_freq=4, connection_density=0.5)
        self.current_level_name = "dungeon_floor"
        self.max_level = 20
        self.record = record
        self.observations = []
        self.timestamps = []
        self.frame_counter = 0
        self.count = 0
        self.fps = 60
        if self.record:
            self.recording_thread = threading.Thread(target=self._record)
            self.record_dir = generate_result_folder_name()
            os.makedirs(f"videos/{self.record_dir}", exist_ok=False)
        else:
            self.recording_thread = None
            self.record_dir = None
        self._init_socket()
        self._start_socket()

    def _init_socket(self):
        logger.info("Initializing socket.")
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.socket.bind((HOSTNAME, PORT))
            logger.info(f"Socket bound to {HOSTNAME}:{PORT}.")
        except Exception:
            logger.error(f"There was a problem binding the socket to {HOSTNAME}:{PORT}.")

    def _initialize_attrs(self, data):
        try:
            self.window_pos_x = data["pos_x"]
            self.window_pos_y = data["pos_y"]
            self.window_width = data["width"]
            self.window_height = data["height"]
        except KeyError:
            self.window_pos_x = 400
            self.window_pos_y = 400
            self.window_width = 800
            self.window_height = 600

        self.monitor = {"top": self.window_pos_y, "left": self.window_pos_x, "width": self.window_width,
                        "height": self.window_height}
        response = {"type": "config", "message": "ALL_SET", "record_dir": self.record_dir}

        return response

    def _record(self):
        print("Recording started...")
        assert self.monitor, "Initialize server first!"
        while not self.disconnect_requested:
            with mss.mss() as sct:
                self.observations.append(np.array(sct.grab(self.monitor)))
                self.timestamps.append(datetime.datetime.now().timestamp())
            time.sleep(1 / self.fps)

    def _start_socket(self):
        self.socket.listen()
        self.connection, address = self.socket.accept()
        logger.info(f"Connected to {address}")
        success = self._three_way_handshake()
        if success == "OK":
            self._listen()
        else:
            logger.error("Something went wrong while establishing the connection.")

    def _three_way_handshake(self):
        message = self.connection.recv(1024)
        payload = json.loads(message.decode("utf-8"))
        if payload["message"] == "SYN":
            response = json.dumps({"type": "handshake", "message": "SYN/ACK"}).encode("utf-8")
            self.connection.sendall(response)
        else:
            logger.error("Received something different from SYN message.")
            return "FAILED"
        message = self.connection.recv(1024)
        payload = json.loads(message.decode("utf-8"))
        if not payload["message"] == "ACK":
            logging.error("Received something different from ACK message.")
            return "FAILED"

        return "OK"

    def _listen(self):
        while not self.disconnect_requested:
            message = self.connection.recv(100000)
            if message:
                response = self._handle_message(message)
                self.connection.sendall(response)
        # When shutting down, wait for confirmation from Godot
        self.connection.recv(100000)
        if self.record:
            self._store_video()

    def _handle_message(self, message):
        payload = json.loads(message.decode("utf-8"))
        if payload["type"] == "data_request":
            response = self._handle_data_request(payload)
        elif payload["type"] == "completion":
            response = self._handle_completion_request(payload)
        elif payload["type"] == "config":
            response = self._handle_config_request(payload)
        else:
            response = json.dumps({"type": "failure", "message": "NONE"}).encode('utf-8')

        return response

    def _handle_data_request(self, request):
        if request["message"] == "FLOOR_DATA_REQUEST":
            response = self.get_maze_map()
        elif request["message"] == "ACTION_REQUEST":
            response = self.get_agent_action()
        elif request["message"] == "LEVEL_NAME_REQUEST":
            response = self.get_current_floor_name()

        return json.dumps(response).encode('utf-8')

    def _handle_completion_request(self, request):
        if request["message"] == "SHUTDOWN":
            self.disconnect_requested = True
            response = {"type": "completion", "message": "SHUTDOWN_COMPLETE"}
        elif request["message"] == "NEW_LEVEL":
            _ = self.dungeon_master.increment_level()
            current_level = self.dungeon_master.get_current_level()
            if current_level <= self.max_level:
                response = {"type": "completion", "message": "NEW_LEVEL_CREATED", "level": str(current_level)}
            else:
                response = {"type": "completion", "message": "MAX_LEVEL_REACHED", "level": str(current_level)}
                self.disconnect_requested = True
        else:
            response = {"type": "completion", "message": "UNKNOWN"}

        return json.dumps(response).encode('utf-8')

    def _handle_config_request(self, request):
        if request["message"] == "SET_WINDOW_COORDS":
            response = self._initialize_attrs(request)
        elif request["message"] == "START_RECORDING":
            self.recording_thread.start()
            response = {"type": "config", "message": "RECORD_OK"}
        else:
            response = {"type": "config", "message": "UNKNOWN"}

        return json.dumps(response).encode('utf-8')

    def _store_video(self):
        """video = cv2.VideoWriter("videos/test.avi", cv2.VideoWriter_fourcc(*'DIVX'), 24, (800, 600))
        for image in self.observations:
            video.write(image)"""
        images = []
        for image in tqdm(self.observations, desc="Saving observations..."):
            frame = np.array(image, dtype=np.uint8)
            frame = np.flip(frame[:, :, :3], 2)
            frame = np.array(Image.fromarray(frame, mode="RGB").resize(size=(160, 120)))
            images.append(frame)
        images = np.asarray(images)

        n, height, width, channels = images.shape
        process = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
            .filter('fps', round='up')
            .output(f"videos/{self.record_dir}/video.mp4", pix_fmt='yuv420p', vcodec='libx264')
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )
        for frame in images:
            process.stdin.write(
                frame
                .astype(np.uint8)
                .tobytes()
            )
        process.stdin.close()
        process.wait()

        #plt.imshow(images[135])
        #plt.show()
        #plt.imshow(self.observations[135])
        #plt.show()
        with open(f"videos/{self.record_dir}/obs_timestamps.csv", "a+") as f:
            for t in self.timestamps:
                f.write(f"{t}\n")
        #np.savez_compressed(f"videos/{self.record_dir}/test.npz", observations=images, timestamps=self.timestamps)

    def get_maze_map(self):
        maze_graph, maze_grid = self.dungeon_master.get_current_floor()
        connections = self.dungeon_master.get_connections()
        connects = np.empty(shape=maze_grid.shape, dtype=object)
        orients = np.zeros(shape=maze_grid.shape)
        for i, j in np.ndindex(maze_grid.shape):
            if maze_grid[i, j] == 0:
                connects[i, j] = "empty"
            elif maze_grid[i, j] == 1:
                keys = list(connections[(i, j)].keys())
                if len(keys) == 2:
                    if set(keys) == {"north", "south"} or set(keys) == {"west", "east"}:
                        connects[i, j] = f"room_2I"
                    else:
                        connects[i, j] = f"room_2L"
                elif len(keys) == 0:
                    connects[i, j] = "boss_room"
                else:
                    connects[i, j] = f"room_{len(list(connections[(i, j)].keys()))}"
            else:
                corridor_name = self.find_corridor_type(list(connections[(i, j)]))
                connects[i, j] = corridor_name
            orients[i, j] = self.find_orientations(connections, i, j)

        connections_string = ""
        for i in range(connects.shape[0]):
            for j in range(connects.shape[1]):
                connections_string += f"{connects[i, j]},"
            connections_string = connections_string[:-1]
            connections_string += "\n"

        orientations_string = ""
        for i in range(orients.shape[0]):
            for j in range(orients.shape[1]):
                orientations_string += f"{int(orients[i, j])},"
            orientations_string = orientations_string[:-1]
            orientations_string += "\n"

        goal_position, agent_position = self.dungeon_master.choose_goal_and_agent_positions()

        maze_floor_map = self.dungeon_master.build_floor_map(goal_pos=goal_position, agent_pos=agent_position)
        plt.imshow(maze_floor_map)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        data = {
            "type": "floor_data_response",
            "connections": connections_string,
            "orientations": orientations_string,
            "player_position": str(agent_position).strip("()").replace(" ", ""),
            "goal_position": str(goal_position).strip("()").replace(" ", ""),
            "level": self.dungeon_master.get_current_level()
        }
        """data = {
            "connections": "junction_T,junction_L,dead_end,corridor\njunction_T,junction_L,dead_end,corridor\njunction_T,junction_L,dead_end,corridor\njunction_T,junction_L,dead_end,corridor\n",
            "orientations": "0,0,0,0\n1,1,1,1\n2,2,2,2\n3,3,3,3\n"
        }"""

        return data

    def get_agent_action(self):
        #img = plt.imread("C:/Users/feder/GodotProjects/MiniCrawl/tmp.jpg")
        self.count += 1
        data = {
            "type": "action_response",
            "action": str(np.random.randint(1, 5))
        }
        return data

    def get_new_floor(self):
        level_name = self.dungeon_master.increment_level()
        data = {
            "type": "level_name_response",
            "message": level_name
        }

        return json.dumps(data).encode('utf-8')

    def get_current_floor_name(self):
        level_name = self.dungeon_master.get_current_level_name()
        data = {
            "type": "level_name_response",
            "message": level_name
        }

        return data

    @staticmethod
    def find_orientations(connections, x, y):
        keys = list(connections[(x, y)].keys())
        if len(keys) == 1:
            return ORIENTATIONS[1][keys[0]]
        elif len(keys) == 2:
            for orient in ORIENTATIONS[2].keys():
                if all([k in orient for k in keys]):
                    return ORIENTATIONS[2][orient]
        elif len(keys) == 3:
            for orient in ORIENTATIONS[3].keys():
                if all([k in orient for k in keys]):
                    return ORIENTATIONS[3][orient]
        else:
            return 0

    @staticmethod
    def find_corridor_type(keys):
        if len(keys) == 1:
            return "dead_end"
        elif len(keys) == 2:
            if set(keys) == {"north", "south"} or set(keys) == {"east", "west"}:
                return "corridor"
            else:
                return "junction_L"
        elif len(keys) == 3:
            return "junction_T"
        else:
            return "junction_X"


if __name__ == '__main__':
    env = GymFlaskServer(record=True)
    print(env.count)
