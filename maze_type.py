import copy
from abc import ABC, abstractmethod

COLOUR_BLACK = [0, 0, 0]
COLOUR_GRAY = [40, 40, 40]


class SizeValidator:
    @staticmethod
    def is_even(number: int) -> bool:
        if number % 2 == 0:
            return True
        return False

    def _correct_size(self, value: int) -> int:
        if self.is_even(value):
            return value - 1
        else:
            return value

    def validate_size(self, dim: int) -> int:
        if dim >= 5:
            return self._correct_size(dim)
        else:
            raise ValueError("Maze size needs to be at least 5x5.")


class MazeBuilder(ABC):
    def __init__(self, width: int, height: int):
        self.validator = SizeValidator()
        self.width = width
        self.height = height
        self._maze = None
        self._init_maze()
        self._in_cells: list[tuple[int, int]] = []
        self.build_steps: list[list[list[list[int, int, int]]]]

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        self._width = self.validator.validate_size(value)

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        self._height = self.validator.validate_size(value)

    def _init_maze(self) -> None:
        self._maze = [[COLOUR_GRAY] * self.width for _ in range(self.height)]
        self.build_steps = [copy.deepcopy(self._maze)]

    @abstractmethod
    def build_maze(self):
        pass

    def _save_next_anim_frame(self):
        self.build_steps.append(copy.deepcopy(self._maze))