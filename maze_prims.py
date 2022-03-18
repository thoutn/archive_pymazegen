from maze_mixins import CheckStatusMixin, CellMixin, FrontierMixin
from maze_type import MazeBuilder
from presenter import MazePresenterThickStyle, MazePresenterThinStyle


class RandomisedPrimsMazeBuilder(MazeBuilder, CheckStatusMixin, CellMixin, FrontierMixin):
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
        self._frontier_cells: list[tuple[int, int]] = []
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
                    xx, yy = self._choose_next_cell_from(self._frontier_cells)

                get_neighbours_of_cell(xx, yy)
                if self._paths_to_cell:
                    path_x, path_y = choose_a_neighbour()
                    save_path_to_neighbour(path_x, path_y)

                    solidify_walls()

        x, y = self._choose_start_cell()
        carve_passage(x, y)

        self._add_maze_start_point()
        self._add_maze_end_point()


if __name__ == '__main__':
    # size = input("Enter maze size as \"width\"x\"height\": ")
    # w, h = size.split('x')
    #
    # prims_maze = RandomisedPrimsMazeBuilder(int(w), int(h))
    # prims_maze.build_maze()
    #
    # thick_walls = MazePresenterThickStyle(prims_maze.build_steps, 10)
    # thick_walls.maze_to_img().show()
    #
    # thin_walls = MazePresenterThinStyle(prims_maze.build_steps)
    # thin_walls.maze_to_img().show()

    # thin_walls.maze_to_animation('maze_prims', 120)

    import timeit
    stmt_code = "prims_maze = RandomisedPrimsMazeBuilder(50, 50) \nprims_maze.build_maze()"
    time = timeit.repeat(stmt=stmt_code, setup="from __main__ import RandomisedPrimsMazeBuilder", repeat=5,
                         number=10)
    print(time)
