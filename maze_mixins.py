from enum import IntEnum
from random import randrange, shuffle

from maze_type import COLOUR_GRAY


COLOUR_WHITE = [255, 255, 255]
COLOUR_WHITE2 = [240, 240, 240]
COLOUR_YELLOW = [255, 211, 67]
COLOUR_YELLOW2 = [204, 156, 0]
COLOUR_GREEN = [202, 224, 12]
COLOUR_GREEN2 = [160, 178, 9]
COLOUR_RED = [250, 72, 27]  # [247, 54, 5]
COLOUR_RED2 = [198, 44, 5]
COLOUR_LRED = [240, 128, 128]
COLOUR_LGREEN = [153, 255, 153]  # [128, 255, 128]  # [102, 255, 102]


class Cell(IntEnum):
    START = 0
    END = 1


class CheckStatusMixin:
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


class StartEndMixin:
    def _add_head_and_tail(self, c: Cell) -> None:
        x, y = ((self.width - 1) * c, randrange(1, self.height - 1, 2))
        self._mark_cell(x, y)
        self._save_next_anim_frame()

    def _add_maze_start_point(self):
        self._add_head_and_tail(Cell.START)

    def _add_maze_end_point(self):
        self._add_head_and_tail(Cell.END)


class CellMixin(StartEndMixin):
    def _mark_cell(self, cx, cy):
        self._maze[cy][cx] = COLOUR_WHITE2

    def _save_cell(self, xx, yy):
        self._mark_cell(xx, yy)
        self._in_cells.append((xx, yy))

    def _mark_wall(self, wx, wy):
        self._maze[wy][wx] = COLOUR_GRAY


class FrontierMixin:
    def _choose_start_cell(self) -> tuple[int, int]:
        x, y = (randrange(1, self.width - 1, 2), randrange(1, self.height - 1, 2))
        return x, y

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

    def _choose_next_cell_from(self, container: list[tuple[int, int]]) -> tuple[int, int]:
        return self._choose_a_random_element(container)


class BacktrackMixin(FrontierMixin):
    def _save_path_to_cell(self, xxx, yyy):
        delta_x = (self._in_cells[-1][0] - xxx) // 2
        delta_y = (self._in_cells[-1][1] - yyy) // 2
        self._save_cell(xxx + delta_x, yyy + delta_y)

    def _remove_frontiers(self):
        while self._frontier_cells:
            xx, yy = self._frontier_cells.pop()
            delta_x = (self._in_cells[-2][0] - xx) // 2
            delta_y = (self._in_cells[-2][1] - yy) // 2
            self._mark_wall(xx, yy)
            self._mark_wall(xx + delta_x, yy + delta_y)

    def _mark_backtrack(self, xxx, yyy):
        self._maze[yyy][xxx] = COLOUR_RED
