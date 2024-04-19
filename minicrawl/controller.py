import keyboard
import pynput


ACTIONS_DICT = {
    "a": 0,
    "d": 1,
    "w": 2,
    "s": 3,
    "e": 4,
    "q": 5
}


class BaseController:
    def __init__(self):
        self._last_key = None

    def wait_press(self):
        self._last_key = keyboard.read_key(suppress=True)

        return self._map_key_press()

    def _map_key_press(self):
        try:
            action = ACTIONS_DICT[self._last_key]
        except KeyError:
            action = 6

        return action


class BasePynputController:
    def __init__(self):
        self._last_key = None

    def wait_press(self):
        with pynput.keyboard.Events() as events:
            try:
                self._last_key = events.get().key.char
            except AttributeError:
                self._last_key = "x"

        return self._map_key_press()

    def _map_key_press(self):
        try:
            action = ACTIONS_DICT[self._last_key]
        except KeyError:
            action = 6

        return action
