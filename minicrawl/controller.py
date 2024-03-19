import keyboard
import mouse


class BaseController:
    def __init__(self):
        self._last_key = None

    def wait_press(self):
        self._last_key = keyboard.read_key(suppress=True)

        return self._map_key_press()

    def _map_key_press(self):
        action = 6
        if self._last_key == "a":
            action = 0
        elif self._last_key == "d":
            action = 1
        elif self._last_key == "w":
            action = 2
        elif self._last_key == "s":
            action = 3
        elif self._last_key == "e":
            action = 4
        elif self._last_key == "q":
            action = 5

        return action


# TODO: check mouse/keyboard docs
class MouseKeyboardController(BaseController):
    def __init__(self):
        super().__init__()
        self._last_mouse_pos = None

    def wait_press(self):
        self._last_mouse_pos = mouse.get_position()
        self._last_key = keyboard.read_key(suppress=True)

    def _map_key_press(self):
        action = 6
        if self._last_key == "a":
            action = 0
        elif self._last_key == "d":
            action = 1
        elif self._last_key == "w":
            action = 2
        elif self._last_key == "s":
            action = 3
        elif self._last_key == "e":
            action = 4
        elif self._last_key == "q":
            action = 5

        return action