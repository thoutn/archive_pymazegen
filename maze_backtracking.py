from enum import IntEnum

from maze_mixins import CheckStatusMixin, CellMixin, BacktrackMixin, COLOUR_WHITE2
from maze_type import MazeBuilder
from presenter import MazePresenterThickStyle, MazePresenterThinStyle


COLOUR_LRED = [240, 128, 128]


class Axis(IntEnum):
    X = 0
    Y = 1


class OldRecursiveBacktrackerMazeBuilder(MazeBuilder, CheckStatusMixin, CellMixin, BacktrackMixin):
    """
    The Recursive Backtracker algorithm is based on the Depth First Search (DFS) technique.

    Uses two randomisation steps:
        - 1st randomisation occurs when a random cell is chosen as the start,
        - 2nd randomisation occurs when choosing a frontier cell (child vertex) to carve into from the current cell.
    """
    def __init__(self, width: int, height: int):
        super().__init__(width, height)
        self._frontier_cells: list[tuple[int, int]] = []
        self._stack: list[tuple[int, int]] = []

    def build_maze(self):
        def take_cell_from_stack():
            xxx, yyy = self._stack.pop()
            self._mark_backtrack(xxx, yyy)
            return xxx, yyy

        def carve_passage(xx: int, yy: int) -> None:
            while self._stack or not self._in_cells:
                self._save_cell(xx, yy)
                self._save_next_anim_frame()

                self._save_new_frontiers(xx, yy)
                self._save_next_anim_frame()

                if self._frontier_cells:
                    self._stack.append((xx, yy))
                    xx, yy = self._choose_next_cell_from(self._frontier_cells)
                    self._save_path_to_cell(xx, yy)
                    self._remove_frontiers()
                else:
                    xx, yy = take_cell_from_stack()
                    self._save_next_anim_frame()

            self._mark_cell(xx, yy)
            self._save_next_anim_frame()

        x, y = self._choose_start_cell()
        carve_passage(x, y)

        self._add_maze_start_point()
        self._add_maze_end_point()


class RecursiveBacktrackingMazeBuilder(MazeBuilder, CheckStatusMixin, CellMixin, BacktrackMixin):
    """
    The Recursive Backtracking algorithm is based on the Depth First Search (DFS) technique.

    Uses two randomisation steps:
        - 1st randomisation occurs when a random cell is chosen as the start,
        - 2nd randomisation occurs when choosing a frontier cell (child vertex) to carve into from the current cell.
    """
    def __init__(self, width: int, height: int):
        super().__init__(width, height)
        self._frontier_cells: list[tuple[int, int]] = []
        self._stack: list[tuple[int, int]] = []
        self._is_backtracking: bool = False

    def _mark_frontier(self, xx, yy, delta_x, delta_y):
        if not self._is_marked(xx + 2 * delta_x, yy + 2 * delta_y, self._frontier_cells):
            self._frontier_cells.append((xx + 2 * delta_x, yy + 2 * delta_y))

    def build_maze(self):
        def mark_backtrack(xx, yy):
            self._maze[yy][xx] = COLOUR_WHITE2
            self._maze[get_path_coord_to_cell(yy, Axis.Y)][get_path_coord_to_cell(xx, Axis.X)] = COLOUR_WHITE2

        def mark_track(xx, yy):
            self._maze[yy][xx] = COLOUR_LRED

        def save_cell(xx, yy):
            self._stack.append((xx, yy))
            self._in_cells.append((xx, yy))
            mark_track(xx, yy)
            self._save_next_anim_frame()

        def get_path_coord_to_cell(coord, axis=Axis.X):
            delta_c = (self._stack[-1][axis] - coord) // 2
            return coord + delta_c

        def save_forward_track(xx, yy):
            mark_track(get_path_coord_to_cell(xx, Axis.X), get_path_coord_to_cell(yy, Axis.Y))
            save_cell(xx, yy)

        def carve_passage(xx: int, yy: int) -> None:
            while self._stack:
                self._save_new_frontiers(xx, yy)

                if self._frontier_cells:
                    if self._is_backtracking:
                        self._stack.append((xx, yy))
                        self._is_backtracking = False

                    xx, yy = self._choose_next_cell_from(self._frontier_cells)
                    self._frontier_cells.clear()

                    save_forward_track(xx, yy)
                else:
                    if self._is_backtracking:
                        mark_backtrack(xx, yy)
                    else:
                        self._is_backtracking = True

                    xx, yy = self._stack.pop()
                    self._save_next_anim_frame()

            self._mark_cell(xx, yy)
            self._save_next_anim_frame()

        x, y = self._choose_start_cell()
        save_cell(x, y)

        carve_passage(x, y)

        self._add_maze_start_point()
        self._add_maze_end_point()


if __name__ == '__main__':
    size = input("Enter maze size as \"width\"x\"height\": ")
    w, h = size.split('x')

    backtracking_maze = RecursiveBacktrackingMazeBuilder(int(w), int(h))
    backtracking_maze.build_maze()

    thick_walls = MazePresenterThickStyle(backtracking_maze.build_steps, 10)
    thick_walls.maze_to_img().show()

    thin_walls = MazePresenterThinStyle(backtracking_maze.build_steps)
    thin_walls.maze_to_img().show()

    thin_walls.maze_to_animation('maze_backtracking', 120)

    # import timeit
    #
    # stmt_code = "backtracker_maze = OldRecursiveBacktrackerMazeBuilder(50, 50) \nbacktracker_maze.build_maze()"
    # time = timeit.repeat(stmt=stmt_code, setup="from __main__ import OldRecursiveBacktrackerMazeBuilder", repeat=5,
    #                      number=10)
    # print(time)
    #
    # stmt_code = "backtracking_maze = RecursiveBacktrackingMazeBuilder(50, 50) \backtracking_maze.build_maze()"
    # time = timeit.repeat(stmt=stmt_code, setup="from __main__ import RecursiveBacktrackingMazeBuilder", repeat=5,
    #                      number=10)
    # print(time)
