from random import shuffle
from enum import IntEnum

from maze_mixins import CheckStatusMixin, CellMixin, BacktrackMixin, COLOUR_LGREEN
from maze_type import MazeBuilder
from presenter import MazePresenterThickStyle, MazePresenterThinStyle


class Axis(IntEnum):
    X = 0
    Y = 1


class OldHuntAndKillMazeBuilder(MazeBuilder, CheckStatusMixin, CellMixin, BacktrackMixin):
    """
    The Hunt-and-Kill algorithm is a modification of the Backtracker algorithm.

    Uses two randomisation steps:
        - 1st randomisation occurs when a random cell is chosen as the start,
        - 2nd randomisation occurs when choosing an unvisited neighbour (vertex) of already visited cells.
    """
    def __init__(self, width: int, height: int):
        super().__init__(width, height)
        self._frontier_cells: list[tuple[int, int]] = []
        self._unvisited_neighbours: list[tuple[tuple[int, int], tuple[int, int]]] = []

    def build_maze(self):
        def drop_visited_cells():
            for unvisited in self._unvisited_neighbours[:]:
                if unvisited[0] in self._in_cells:
                    self._unvisited_neighbours.remove(unvisited)

        def take_cell_from_unvisited_neighbours():
            drop_visited_cells()
            shuffle(self._unvisited_neighbours)
            if not self._unvisited_neighbours:
                return None, None
            xxx, yyy = self._unvisited_neighbours.pop()[1]
            self._mark_backtrack(xxx, yyy)
            return xxx, yyy

        def save_unvisited_neighbours():
            for frontier in self._frontier_cells:
                self._unvisited_neighbours.append((frontier, self._in_cells[-2]))

        def carve_passage(xx: int, yy: int) -> None:
            while self._unvisited_neighbours or not self._in_cells:
                self._save_cell(xx, yy)
                self._save_next_anim_frame()

                self._save_new_frontiers(xx, yy)
                self._save_next_anim_frame()

                if self._frontier_cells:
                    xx, yy = self._choose_next_cell_from(self._frontier_cells)
                    self._save_path_to_cell(xx, yy)
                    save_unvisited_neighbours()
                    self._remove_frontiers()
                else:
                    xx, yy = take_cell_from_unvisited_neighbours()
                    if not xx:
                        break
                    self._save_next_anim_frame()

            if xx:
                self._mark_cell(xx, yy)
                self._save_next_anim_frame()

        x, y = self._choose_start_cell()
        carve_passage(x, y)

        self._add_maze_start_point()
        self._add_maze_end_point()


class HuntAndKillMazeBuilder(MazeBuilder, CheckStatusMixin, CellMixin, BacktrackMixin):
    """
    The Hunt-and-Kill algorithm is a modification of the Backtracker algorithm.

    Uses two randomisation steps:
        - 1st randomisation occurs when a random cell is chosen as the start,
        - 2nd randomisation occurs when choosing an unvisited neighbour (vertex) of already visited cells.
    """
    def __init__(self, width: int, height: int):
        super().__init__(width, height)
        self._frontier_cells: list[tuple[int, int]] = []
        self._unvisited_neighbours: list[tuple[tuple[int, int], tuple[int, int]]] = []

    def _mark_frontier(self, xx, yy, delta_x, delta_y):
        if not self._is_marked(xx + 2 * delta_x, yy + 2 * delta_y, self._frontier_cells):
            self._frontier_cells.append((xx + 2 * delta_x, yy + 2 * delta_y))

    def build_maze(self):
        def get_path_coord_to_cell(coord, axis=Axis.X):
            delta_c = (self._in_cells[-1][axis] - coord) // 2
            return coord + delta_c

        def mark_track(xx, yy):
            self._maze[yy][xx] = COLOUR_LGREEN

        def save_forward_track(xx, yy):
            self._mark_cell(get_path_coord_to_cell(xx, Axis.X), get_path_coord_to_cell(yy, Axis.Y))
            mark_track(xx, yy)
            self._save_next_anim_frame()

        def drop_visited_cells():
            for unvisited in self._unvisited_neighbours[:]:
                if unvisited[0] in self._in_cells:
                    self._unvisited_neighbours.remove(unvisited)

        def take_cell_from_unvisited_neighbours():
            drop_visited_cells()
            shuffle(self._unvisited_neighbours)
            if not self._unvisited_neighbours:
                return None, None
            xxx, yyy = self._unvisited_neighbours.pop()[1]
            # self._mark_backtrack(xxx, yyy)
            mark_track(xxx, yyy)
            return xxx, yyy

        def save_unvisited_neighbours():
            for frontier in self._frontier_cells:
                # self._unvisited_neighbours.append((frontier, self._in_cells[-2]))
                self._unvisited_neighbours.append((frontier, self._in_cells[-1]))

        def carve_passage(xx: int, yy: int) -> None:
            while self._unvisited_neighbours or not self._in_cells:
                self._save_cell(xx, yy)
                self._save_next_anim_frame()

                self._save_new_frontiers(xx, yy)
                # self._save_next_anim_frame()

                if self._frontier_cells:
                    xx, yy = self._choose_next_cell_from(self._frontier_cells)
                    # self._save_path_to_cell(xx, yy)
                    save_forward_track(xx, yy)
                    save_unvisited_neighbours()
                    # self._remove_frontiers()
                    self._frontier_cells.clear()
                else:
                    xx, yy = take_cell_from_unvisited_neighbours()
                    if not xx:
                        break
                    self._save_next_anim_frame()

            if xx:
                self._mark_cell(xx, yy)
                self._save_next_anim_frame()

        x, y = self._choose_start_cell()
        carve_passage(x, y)

        self._add_maze_start_point()
        self._add_maze_end_point()


if __name__ == '__main__':
    # size = input("Enter maze size as \"width\"x\"height\": ")
    # w, h = size.split('x')
    #
    # huntkill_maze = HuntAndKillMazeBuilder(int(w), int(h))
    # huntkill_maze.build_maze()
    #
    # thin_walls = MazePresenterThinStyle(huntkill_maze.build_steps)
    # thin_walls.maze_to_img().show()
    # #
    # thin_walls.maze_to_animation('maze_huntkill', 120)

    import timeit

    stmt_code = "huntkill_maze = OldHuntAndKillMazeBuilder(50, 50) \nhuntkill_maze.build_maze()"
    time = timeit.repeat(stmt=stmt_code, setup="from __main__ import OldHuntAndKillMazeBuilder", repeat=5,
                         number=10)
    print(time)

    stmt_code = "huntkill_maze = HuntAndKillMazeBuilder(50, 50) \nhuntkill_maze.build_maze()"
    time = timeit.repeat(stmt=stmt_code, setup="from __main__ import HuntAndKillMazeBuilder", repeat=5,
                         number=10)
    print(time)
