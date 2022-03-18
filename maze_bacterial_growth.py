from random import shuffle

from maze_mixins import CheckStatusMixin, CellMixin, FrontierMixin, COLOUR_YELLOW, COLOUR_YELLOW2
from maze_type import MazeBuilder
from presenter import MazePresenterThickStyle, MazePresenterThinStyle


class BacterialGrowthMazeBuilder(MazeBuilder, CheckStatusMixin, CellMixin, FrontierMixin):
    """
        The Bacterial Growth algorithm is a modification of Prim's algorithm. Rather than adding one cell in each step,
        this algorithm carves into one random unvisited neighbour - if available - per each visited cell.

        Uses two randomisation steps:
            - 1st randomisation occurs when a random cell is chosen as start,
            - 2nd randomisation occurs when a random cell is chosen from the set of unvisited neighbours for each
              visited cell.
        """
    def __init__(self, width: int, height: int):
        super().__init__(width, height)
        self._frontier_cells: list[tuple[int, int]] = []
        self._unvisited_neighbours: list[tuple[tuple[int, int], tuple[int, int]]] = []
        self._visited_with_unvisited_neighbours: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        self._visited_cells: list[tuple[int, int]] = []

    # @Override
    def _mark_frontier(self, xx, yy, delta_x, delta_y):
        unvisited = ((xx + 2 * delta_x, yy + 2 * delta_y), (xx, yy))
        self._visited_with_unvisited_neighbours.add(unvisited)

        self._maze[yy + 2 * delta_y][xx + 2 * delta_x] = COLOUR_YELLOW
        self._maze[yy + delta_y][xx + delta_x] = COLOUR_YELLOW2

    def _save_path_to_cell(self, xxx, yyy):
        delta_x = (self._in_cells[-1][0] - xxx) // 2
        delta_y = (self._in_cells[-1][1] - yyy) // 2
        self._save_cell(xxx + delta_x, yyy + delta_y)

    def _remove_frontiers(self, ix, iy):
        while self._unvisited_neighbours:
            xx, yy = self._unvisited_neighbours.pop()[0]
            delta_x = (ix - xx) // 2
            delta_y = (iy - yy) // 2
            self._mark_wall(xx, yy)
            self._mark_wall(xx + delta_x, yy + delta_y)

    def build_maze(self):
        def remove_path(frontier, in_cell):
            fx, fy = frontier
            ix, iy = in_cell
            delta_x = (ix - fx) // 2
            delta_y = (iy - fy) // 2
            self._mark_wall(fx + delta_x, fy + delta_y)

        def drop_visited_cells():
            for unvisited in self._unvisited_neighbours[:]:
                if unvisited[0] in self._in_cells:
                    remove_path(unvisited[0], unvisited[1])
                    self._unvisited_neighbours.remove(unvisited)

        def save_as_visited():
            drop_visited_cells()
            shuffle(self._unvisited_neighbours)
            if not self._unvisited_neighbours:
                return

            (fx, fy), (ix, iy) = self._unvisited_neighbours.pop()
            self._remove_frontiers(ix, iy)

            self._save_cell(fx, fy)
            self._visited_cells.append((fx, fy))
            self._save_path_to_cell(ix, iy)

        def collect_unvisited_neighbours():
            for cell in self._visited_cells[:]:
                start = len(self._visited_with_unvisited_neighbours)

                self._save_new_frontiers(cell[0], cell[1])

                is_new_element_added = len(self._visited_with_unvisited_neighbours) > start
                if not is_new_element_added:
                    self._visited_cells.remove(cell)

        def carve_unvisited_neighbours():
            # shuffle(self._visited_cells)
            for cell in self._visited_cells:
                self._unvisited_neighbours = [un for un in self._visited_with_unvisited_neighbours if un[1] == cell]
                save_as_visited()
            self._visited_with_unvisited_neighbours.clear()

        def carve_passage(xx: int, yy: int) -> None:
            self._save_cell(xx, yy)
            self._visited_cells.append((xx, yy))
            self._save_next_anim_frame()

            while len(self._visited_cells) > 0:
                collect_unvisited_neighbours()
                self._save_next_anim_frame()

                carve_unvisited_neighbours()
                self._save_next_anim_frame()

        x, y = self._choose_start_cell()
        carve_passage(x, y)

        self._add_maze_start_point()
        self._add_maze_end_point()


if __name__ == '__main__':
    size = input("Enter maze size as \"width\"x\"height\": ")
    w, h = size.split('x')

    bacterial_maze = BacterialGrowthMazeBuilder(int(w), int(h))
    bacterial_maze.build_maze()

    thick_walls = MazePresenterThickStyle(bacterial_maze.build_steps, 10)
    thick_walls.maze_to_img().show()

    thin_walls = MazePresenterThinStyle(bacterial_maze.build_steps)
    thin_walls.maze_to_img().show()

    thin_walls.maze_to_animation('maze_bacterial', 120)

    # import timeit
    # stmt_code = "bacterial_maze = BacterialGrowthMazeBuilder(50, 50) \nbacterial_maze.build_maze()"
    # time = timeit.repeat(stmt=stmt_code, setup="from __main__ import BacterialGrowthMazeBuilder", repeat=5,
    #                      number=10)
    # print(time)
