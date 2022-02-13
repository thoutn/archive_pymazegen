import numpy as np
import numpy.typing as npt
from PIL import Image
from pprint import pprint
from random import randrange, shuffle
import copy
from enum import IntEnum
from abc import ABC, abstractmethod


CELL_SIZE = 20
COLOUR_BLACK = [0, 0, 0]
COLOUR_WHITE = [255, 255, 255]
COLOUR_WHITE2 = [240, 240, 240]
COLOUR_YELLOW = [255, 211, 67]
COLOUR_YELLOW2 = [204, 156, 0]
COLOUR_GREEN = [202, 224, 12]
COLOUR_GREEN2 = [160, 178, 9]
COLOUR_GRAY = [40, 40, 40]


class Cell(IntEnum):
    START = 0
    END = 1


def resize_maze(maze_: list, multiple: int = CELL_SIZE) -> np.ndarray:
    def resize_rows(list_: np.ndarray) -> np.ndarray:
        enlarged = [[row] * multiple for row in list_]
        return np.array(enlarged).reshape((len(list_) * multiple, len(list_[0]), -1))

    def resize_columns(list_: list) -> np.ndarray:
        enlarged = [[i] * multiple for column in list_ for i in column]
        return np.array(enlarged).reshape((len(list_), len(list_[0]) * multiple, -1))

    return resize_rows(resize_columns(maze_))


def maze_to_img(maze_: list, size_: int = CELL_SIZE) -> Image:
    maze_ = resize_maze(maze_, size_)
    pil_image = Image.fromarray(np.uint8(maze_)).convert('RGB')
    return pil_image


def maze_to_animation(maze_list: list, file_name: str, size_: int = CELL_SIZE, time_: int = 200) -> None:
    imgs = []
    for maze_ in maze_list:
        imgs.append(maze_to_img(maze_, size_))

    imgs[0].save(file_name + '.gif', save_all=True, append_images=imgs[1:], optimize=False, duration=time_)


class MazeBuilder(ABC):
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self._init_maze()

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        self._width = MazeBuilder.validate_size(value)

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        self._height = MazeBuilder.validate_size(value)

    @staticmethod
    def is_even(number: int) -> bool:
        if number % 2 == 0:
            return True
        return False

    @staticmethod
    def correct_size(value: int) -> int:
        if MazeBuilder.is_even(value):
            return value - 1
        else:
            return value

    @staticmethod
    def validate_size(dim: int) -> tuple[int, int]:
        if dim >= 5:
            return MazeBuilder.correct_size(dim)
        else:
            raise ValueError("Maze size needs to be at least 5x5.")

    def _init_maze(self) -> None:
        self._maze = [[COLOUR_GRAY] * self.width for _ in range(self.height)]

    @abstractmethod
    def build_maze(self):
        pass


class RandomisedPrimsMazeBuilder(MazeBuilder):
    def __init__(self, width: int, height: int):
        super().__init__(width, height)
        self._in_cells: list[tuple[int, int]] = []
        self._frontier_cells: list[tuple[int, int]] = []
        self._paths_to_next_cell: list[tuple[int, int]] = []
        self.build_steps: list[list[list[list[int, int, int]]]] = [copy.deepcopy(self._maze)]

    def build_maze(self) -> None:
        def is_carved(xxx, yyy):
            if (xxx, yyy) in self._in_cells:
                return True
            return False

        def is_near_right_edge(pos):
            return pos == self.width - 2

        def is_near_left_edge(pos):
            return pos == 1

        def is_near_bottom_edge(pos):
            return pos == self.height - 2

        def is_near_top_edge(pos):
            return pos == 1

        def is_marked(xxx, yyy):
            if (xxx, yyy) in self._frontier_cells:
                return True
            return False

        def save_frontier(xxx, yyy, delta_x, delta_y):
            if not is_marked(xxx + 2*delta_x, yyy + 2*delta_y):
                self._frontier_cells.append((xxx + 2*delta_x, yyy + 2*delta_y))
                self._maze[yyy + 2*delta_y][xxx + 2*delta_x] = COLOUR_YELLOW
            self._maze[yyy + delta_y][xxx + delta_x] = COLOUR_YELLOW2

        def mark_cell(cx, cy):
            self._maze[cy][cx] = COLOUR_WHITE2

        def save_cell(xxx, yyy):
            mark_cell(xxx, yyy)
            self._in_cells.append((xxx, yyy))

        def save_next_anim_frame():
            self.build_steps.append(copy.deepcopy(self._maze))

        def mark_new_frontiers(fx, fy):
            if not is_near_right_edge(fx) and not is_carved(fx + 2, fy): save_frontier(fx, fy, 1, 0)
            if not is_near_left_edge(fx) and not is_carved(fx - 2, fy): save_frontier(fx, fy, -1, 0)
            if not is_near_bottom_edge(fy) and not is_carved(fx, fy + 2): save_frontier(fx, fy, 0, 1)
            if not is_near_top_edge(fy) and not is_carved(fx, fy - 2): save_frontier(fx, fy, 0, -1)

        def choose_a_random_element(list_: list[tuple[int, int]]) -> tuple[int, int]:
            shuffle(list_)
            return list_.pop()

        def choose_next_cell_from_frontiers() -> tuple[int, int]:
            return choose_a_random_element(self._frontier_cells)

        def get_neighbours_of_next_cell(xx, yy):
            if not is_near_right_edge(xx) and is_carved(xx + 2, yy): self._paths_to_next_cell.append((xx + 1, yy))
            if not is_near_left_edge(xx) and is_carved(xx - 2, yy): self._paths_to_next_cell.append((xx - 1, yy))
            if not is_near_bottom_edge(yy) and is_carved(xx, yy + 2): self._paths_to_next_cell.append((xx, yy + 1))
            if not is_near_top_edge(yy) and is_carved(xx, yy - 2): self._paths_to_next_cell.append((xx, yy - 1))

        def choose_a_neighbour() -> tuple[int, int]:
            return choose_a_random_element(self._paths_to_next_cell)

        def add_path_to_neighbour(px: int, py: int) -> None:
            mark_cell(px, py)

        def solidify_walls():
            while self._paths_to_next_cell:
                path_x, path_y = self._paths_to_next_cell.pop()
                self._maze[path_y][path_x] = COLOUR_GRAY

        def add_head_and_tail(c: Cell) -> None:
            while True:
                x, y = ((self.width - 1) * c, randrange(1, self.height - 1))
                if is_carved(x + 1 - 2*c, y):
                    mark_cell(x, y)
                    save_next_anim_frame()
                    break

        def add_maze_start_point():
            add_head_and_tail(Cell.START)

        def add_maze_end_point():
            add_head_and_tail(Cell.END)

        def carve_passage(xx: int, yy: int) -> None:
            while True:
                save_cell(xx, yy)
                save_next_anim_frame()

                mark_new_frontiers(xx, yy)
                save_next_anim_frame()

                if not self._frontier_cells:
                    break
                else:
                    xx, yy = choose_next_cell_from_frontiers()

                get_neighbours_of_next_cell(xx, yy)
                if self._paths_to_next_cell:
                    path_x, path_y = choose_a_neighbour()
                    add_path_to_neighbour(path_x, path_y)

                    solidify_walls()

        x, y = (randrange(1, self.width - 1, 2), randrange(1, self.height - 1, 2))
        carve_passage(x, y)

        add_maze_start_point()
        add_maze_end_point()


if __name__ == '__main__':
    size = input("Enter maze size as \"width\"x\"height\": ")
    w, h = size.split('x')

    prims_maze = RandomisedPrimsMazeBuilder(int(w), int(h))
    prims_maze.build_maze()

    maze_to_img(prims_maze.build_steps[-1], 10).show()
    maze_to_animation(prims_maze.build_steps, 'maze_big10', 10, 120)
