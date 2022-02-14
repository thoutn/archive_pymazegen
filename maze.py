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
COLOUR_RED = [250, 72, 27]  # [247, 54, 5]
COLOUR_RED2 = [198, 44, 5]
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
        self._in_cells: list[tuple[int, int]] = []
        self.build_steps: list[list[list[list[int, int, int]]]] = [copy.deepcopy(self._maze)]

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

    def _is_carved(self, xxx, yyy):
        if (xxx, yyy) in self._in_cells:
            return True
        return False

    def _mark_cell(self, cx, cy):
        self._maze[cy][cx] = COLOUR_WHITE2

    def _save_cell(self, xxx, yyy):
        self._mark_cell(xxx, yyy)
        self._in_cells.append((xxx, yyy))

    def _mark_wall(self, wx, wy):
        self._maze[wy][wx] = COLOUR_GRAY

    def _add_head_and_tail(self, c: Cell) -> None:
        while True:
            x, y = ((self.width - 1) * c, randrange(1, self.height - 1))
            if self._is_carved(x + 1 - 2 * c, y):
                self._mark_cell(x, y)
                self._save_next_anim_frame()
                break

    def _add_maze_start_point(self):
        self._add_head_and_tail(Cell.START)

    def _add_maze_end_point(self):
        self._add_head_and_tail(Cell.END)

    def _save_next_anim_frame(self):
        self.build_steps.append(copy.deepcopy(self._maze))

    @abstractmethod
    def build_maze(self):
        pass


class RandomisedPrimsMazeBuilder(MazeBuilder):
    def __init__(self, width: int, height: int):
        super().__init__(width, height)
        self._in_cells: list[tuple[int, int]] = []
        self._frontier_cells: list[tuple[int, int]] = []
        self._paths_to_next_cell: list[tuple[int, int]] = []

    def build_maze(self) -> None:
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

        def mark_frontier(xxx, yyy, delta_x, delta_y):
            if not is_marked(xxx + 2*delta_x, yyy + 2*delta_y):
                self._frontier_cells.append((xxx + 2*delta_x, yyy + 2*delta_y))
                self._maze[yyy + 2*delta_y][xxx + 2*delta_x] = COLOUR_YELLOW
            self._maze[yyy + delta_y][xxx + delta_x] = COLOUR_YELLOW2

        def save_new_frontiers(fx, fy):
            if not is_near_right_edge(fx) and not self._is_carved(fx + 2, fy): mark_frontier(fx, fy, 1, 0)
            if not is_near_left_edge(fx) and not self._is_carved(fx - 2, fy): mark_frontier(fx, fy, -1, 0)
            if not is_near_bottom_edge(fy) and not self._is_carved(fx, fy + 2): mark_frontier(fx, fy, 0, 1)
            if not is_near_top_edge(fy) and not self._is_carved(fx, fy - 2): mark_frontier(fx, fy, 0, -1)

        def choose_a_random_element(list_: list[tuple[int, int]]) -> tuple[int, int]:
            shuffle(list_)
            return list_.pop()

        def choose_next_cell_from_frontiers() -> tuple[int, int]:
            return choose_a_random_element(self._frontier_cells)

        def get_neighbours_of_next_cell(xx, yy):
            if not is_near_right_edge(xx) and self._is_carved(xx + 2, yy): self._paths_to_next_cell.append((xx + 1, yy))
            if not is_near_left_edge(xx) and self._is_carved(xx - 2, yy): self._paths_to_next_cell.append((xx - 1, yy))
            if not is_near_bottom_edge(yy) and self._is_carved(xx, yy + 2): self._paths_to_next_cell.append((xx, yy + 1))
            if not is_near_top_edge(yy) and self._is_carved(xx, yy - 2): self._paths_to_next_cell.append((xx, yy - 1))

        def choose_a_neighbour() -> tuple[int, int]:
            return choose_a_random_element(self._paths_to_next_cell)

        def add_path_to_neighbour(px: int, py: int) -> None:
            self._mark_cell(px, py)

        def solidify_walls():
            while self._paths_to_next_cell:
                path_x, path_y = self._paths_to_next_cell.pop()
                self._mark_wall(path_x, path_y)

        def carve_passage(xx: int, yy: int) -> None:
            while True:
                self._save_cell(xx, yy)
                self._save_next_anim_frame()

                save_new_frontiers(xx, yy)
                self._save_next_anim_frame()

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

        self._add_maze_start_point()
        self._add_maze_end_point()


class RandomisedKruskalsMazeBuilder(MazeBuilder):
    def __init__(self, width: int, height: int):
        super().__init__(width, height)
        self._tree_sets = [{(x, y), } for y in range(1, self.height - 1, 2) for x in range(1, self.width - 1, 2)]
        self._hor_walls = [(x, y) for y in range(1, self.height - 1, 2) for x in range(2, self.width - 2, 2)]
        self._ver_walls = [(x, y) for y in range(2, self.height - 2, 2) for x in range(1, self.height - 1, 2)]
        self._walls = self._hor_walls + self._ver_walls

    def build_maze(self) -> None:
        def mark_cells_at_wall(w, c1, c2):
            self._maze[c1[1]][c1[0]] = COLOUR_RED
            self._maze[c2[1]][c2[0]] = COLOUR_RED
            self._maze[w[1]][w[0]] = COLOUR_RED2

        def choose_wall() -> tuple[int, int]:
            shuffle(self._walls)
            wall_x, wall_y = self._walls.pop()
            return wall_x, wall_y

        def get_cells_near_wall(xx: int, yy: int) -> tuple[tuple[int, int], tuple[int, int]]:
            if xx % 2 == 0:
                c1 = (xx - 1, yy)
                c2 = (xx + 1, yy)
            else:
                c1 = (xx, yy - 1)
                c2 = (xx, yy + 1)

            mark_cells_at_wall((xx, yy), c1, c2)
            return c1, c2

        def cells_in_same_tree(c1, c2):
            for set_ in self._tree_sets:
                if c1 in set_:
                    if c2 in set_:
                        return True
                    else:
                        return False

        def join_tree_sets(c1, c2):
            for i, set_ in enumerate(self._tree_sets):
                if c1 in set_:
                    i1 = i
                    break

            for i, set_ in enumerate(self._tree_sets):
                if c2 in set_:
                    i2 = i
                    break

            set_ = self._tree_sets[i2]
            self._tree_sets[i1] = self._tree_sets[i1] | set_
            self._tree_sets.remove(set_)

        def join_cells(c1, c2):
            self._save_cell(c1[0], c1[1])
            self._save_cell(c2[0], c2[1])
            if c1[0] == c2[0]:
                self._save_cell(c1[0], c1[1] + 1)
            else:
                self._save_cell(c1[0] + 1, c1[1])

            join_tree_sets(c1, c2)

        def mark_cannot_remove(xx, yy):
            self._maze[yy][xx] = COLOUR_YELLOW
            self._save_next_anim_frame()
            self._mark_wall(xx, yy)

        def solidify_wall(c1, c2):
            self._save_cell(c1[0], c1[1])
            self._save_cell(c2[0], c2[1])
            if c1[0] == c2[0]:
                mark_cannot_remove(c1[0], c1[1] + 1)
            else:
                mark_cannot_remove(c1[0] + 1, c1[1])

        def carve_passage() -> None:
            while True:
                if self._walls:
                    x, y = choose_wall()
                else:
                    break

                cell_1, cell_2 = get_cells_near_wall(x, y)
                self._save_next_anim_frame()

                if not cells_in_same_tree(cell_1, cell_2):
                    join_cells(cell_1, cell_2)
                else:
                    solidify_wall(cell_1, cell_2)

        carve_passage()

        self._add_maze_start_point()
        self._add_maze_end_point()

if __name__ == '__main__':
    size = input("Enter maze size as \"width\"x\"height\": ")
    w, h = size.split('x')

    prims_maze = RandomisedPrimsMazeBuilder(int(w), int(h))
    prims_maze.build_maze()

    maze_to_img(prims_maze.build_steps[-1], 10).show()
    maze_to_animation(prims_maze.build_steps, 'maze_prims', 10, 120)

    kruskals_maze = RandomisedKruskalsMazeBuilder(int(w), int(h))
    kruskals_maze.build_maze()

    maze_to_img(kruskals_maze.build_steps[-1], 10).show()
    maze_to_animation(kruskals_maze.build_steps, 'maze_kruskals', 10, 120)