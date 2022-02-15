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
        self._frontier_cells: list[tuple[int, int]] = []
        self.build_steps: list[list[list[list[int, int, int]]]] = [copy.deepcopy(self._maze)]

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        self._width = self._validate_size(value)

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        self._height = self._validate_size(value)

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

    def _validate_size(self, dim: int) -> int:
        if dim >= 5:
            return self._correct_size(dim)
        else:
            raise ValueError("Maze size needs to be at least 5x5.")

    def _init_maze(self) -> None:
        self._maze = [[COLOUR_GRAY] * self.width for _ in range(self.height)]

    @staticmethod
    def _near_start(pos: int) -> bool:
        return pos == 1

    @staticmethod
    def _near_end(pos: int, dim: int) -> bool:
        return pos == dim - 2

    def _is_near_right_edge(self, pos):
        return self._near_end(pos, self.width)

    def _is_near_left_edge(self, pos):
        return self._near_start(pos)

    def _is_near_bottom_edge(self, pos):
        return self._near_end(pos, self.height)

    def _is_near_top_edge(self, pos):
        return self._near_start(pos)

    def _is_marked(self, xx, yy, container: list[tuple[int, int]] = None):
        if not container:
            container = self._in_cells
        return (xx, yy) in container

    def _mark_cell(self, cx, cy):
        self._maze[cy][cx] = COLOUR_WHITE2

    def _save_cell(self, xx, yy):
        self._mark_cell(xx, yy)
        self._in_cells.append((xx, yy))

    def _mark_wall(self, wx, wy):
        self._maze[wy][wx] = COLOUR_GRAY

    def _mark_frontier(self, xx, yy, delta_x, delta_y):
        if not self._is_marked(xx + 2 * delta_x, yy + 2 * delta_y, self._frontier_cells):
            self._frontier_cells.append((xx + 2 * delta_x, yy + 2 * delta_y))
            self._maze[yy + 2 * delta_y][xx + 2 * delta_x] = COLOUR_YELLOW
        self._maze[yy + delta_y][xx + delta_x] = COLOUR_YELLOW2

    def _save_new_frontiers(self, fx, fy):
        if not self._is_near_right_edge(fx) and not self._is_marked(fx + 2, fy):
            self._mark_frontier(fx, fy, 1, 0)
        if not self._is_near_left_edge(fx) and not self._is_marked(fx - 2, fy):
            self._mark_frontier(fx, fy, -1, 0)
        if not self._is_near_bottom_edge(fy) and not self._is_marked(fx, fy + 2):
            self._mark_frontier(fx, fy, 0, 1)
        if not self._is_near_top_edge(fy) and not self._is_marked(fx, fy - 2):
            self._mark_frontier(fx, fy, 0, -1)

    @staticmethod
    def _choose_a_random_element(container: list[tuple[int, int]]) -> tuple[int, int]:
        shuffle(container)
        return container.pop()

    def _choose_next_cell_from_frontiers(self) -> tuple[int, int]:
        return self._choose_a_random_element(self._frontier_cells)

    def _add_head_and_tail(self, c: Cell) -> None:
        while True:
            x, y = ((self.width - 1) * c, randrange(1, self.height - 1))
            if self._is_marked(x + 1 - 2 * c, y, self._in_cells):
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
    """
    Prim's algorithm is a greedy algorithm, which can be used to find a minimum spanning tree within a graph.
    The randomised version does not choose the most minimally weighted edge, it chooses any (random) edge
    directly connecting a vertex within the tree with a vertex outside the tree.

    Uses two randomisation steps:
        - 1st randomisation occurs when a random cell is chosen as start,
        - 2nd randomisation occurs when a random cell is chosen from the set of frontier cells.
    """
    def __init__(self, width: int, height: int):
        super().__init__(width, height)
        self._paths_to_cell: list[tuple[int, int]] = []

    def build_maze(self) -> None:
        def get_neighbours_of_cell(xx, yy):
            if not self._is_near_right_edge(xx) and self._is_marked(xx + 2, yy):
                self._paths_to_cell.append((xx + 1, yy))
            if not self._is_near_left_edge(xx) and self._is_marked(xx - 2, yy):
                self._paths_to_cell.append((xx - 1, yy))
            if not self._is_near_bottom_edge(yy) and self._is_marked(xx, yy + 2):
                self._paths_to_cell.append((xx, yy + 1))
            if not self._is_near_top_edge(yy) and self._is_marked(xx, yy - 2):
                self._paths_to_cell.append((xx, yy - 1))

        def choose_a_neighbour() -> tuple[int, int]:
            return self._choose_a_random_element(self._paths_to_cell)

        def save_path_to_neighbour(px: int, py: int) -> None:
            self._mark_cell(px, py)

        def solidify_walls():
            while self._paths_to_cell:
                path_x, path_y = self._paths_to_cell.pop()
                self._mark_wall(path_x, path_y)

        def carve_passage(xx: int, yy: int) -> None:
            while True:
                self._save_cell(xx, yy)
                self._save_next_anim_frame()

                self._save_new_frontiers(xx, yy)
                self._save_next_anim_frame()

                if not self._frontier_cells:
                    break
                else:
                    xx, yy = self._choose_next_cell_from_frontiers()

                get_neighbours_of_cell(xx, yy)
                if self._paths_to_cell:
                    path_x, path_y = choose_a_neighbour()
                    save_path_to_neighbour(path_x, path_y)

                    solidify_walls()

        x, y = (randrange(1, self.width - 1, 2), randrange(1, self.height - 1, 2))
        carve_passage(x, y)

        self._add_maze_start_point()
        self._add_maze_end_point()


class RandomisedKruskalsMazeBuilder(MazeBuilder):
    """
    Kruskal's algorithm is a greedy algorithm, which operates in the whole graph area creating
    and binding fragmental acyclic trees to each other.

    Uses one randomisation step:
        - walls between two cells are chosen in a random order. Unless there is another path
          between the two cells, the wall is removed.
    """
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

        def mark_unremovable_wall(xx, yy):
            self._maze[yy][xx] = COLOUR_YELLOW
            self._save_next_anim_frame()
            self._mark_wall(xx, yy)

        def solidify_wall(c1, c2):
            self._save_cell(c1[0], c1[1])
            self._save_cell(c2[0], c2[1])
            if c1[0] == c2[0]:
                mark_unremovable_wall(c1[0], c1[1] + 1)
            else:
                mark_unremovable_wall(c1[0] + 1, c1[1])

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


class BacktrackerMazeBuilder(MazeBuilder):
    """
    The Backtracker algorithm is based on the Depth First Search (DFS) technique.

    Uses two randomisation steps:
        - 1st randomisation occurs when a random cell is chosen as the start,
        - 2nd randomisation occurs when choosing a frontier cell (child vertex) to carve into from the current cell.
    """
    def __init__(self, width: int, height: int):
        super().__init__(width, height)
        self._stack = []

    def build_maze(self):
        def choose_start_cell() -> tuple[int, int]:
            cells = [(x, y) for y in range(1, self.height - 1, 2) for x in range(1, self.width - 1, 2)]
            shuffle(cells)
            return cells.pop()

        def save_path_to_cell(xxx, yyy):
            delta_x = (self._stack[-1][0] - xxx) // 2
            delta_y = (self._stack[-1][1] - yyy) // 2
            self._save_cell(xxx + delta_x, yyy + delta_y)

        def remove_frontiers():
            while self._frontier_cells:
                xx, yy = self._frontier_cells.pop()
                delta_x = (self._stack[-1][0] - xx) // 2
                delta_y = (self._stack[-1][1] - yy) // 2
                self._mark_wall(xx, yy)
                self._mark_wall(xx + delta_x, yy + delta_y)

        def mark_backtrack(xxx, yyy):
            self._maze[yyy][xxx] = COLOUR_RED

        def take_cell_from_stack():
            xxx, yyy = self._stack.pop()
            mark_backtrack(xxx, yyy)
            return xxx, yyy

        def carve_passage(xx: int, yy: int) -> None:
            while self._stack or not self._in_cells:
                self._save_cell(xx, yy)
                self._save_next_anim_frame()

                self._save_new_frontiers(xx, yy)
                self._save_next_anim_frame()

                if self._frontier_cells:
                    self._stack.append((xx, yy))
                    xx, yy = self._choose_next_cell_from_frontiers()
                    save_path_to_cell(xx, yy)
                    remove_frontiers()
                else:
                    xx, yy = take_cell_from_stack()
                    self._save_next_anim_frame()

            self._mark_cell(xx, yy)
            self._save_next_anim_frame()

        x, y = choose_start_cell()
        carve_passage(x, y)

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

    backtracker_maze = BacktrackerMazeBuilder(int(w), int(h))
    backtracker_maze.build_maze()

    maze_to_img(backtracker_maze.build_steps[-1], 10).show()
    maze_to_animation(backtracker_maze.build_steps, 'maze_backtracker', 10, 120)