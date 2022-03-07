from random import randrange, shuffle

from maze_mixins import CheckStatusMixin, CellMixin, FrontierMixin, COLOUR_YELLOW, COLOUR_RED, COLOUR_RED2, \
    BacktrackMixin, COLOUR_YELLOW2
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


class RandomisedKruskalsMazeBuilder(MazeBuilder, CheckStatusMixin, CellMixin):
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


class BacktrackerMazeBuilder(MazeBuilder, CheckStatusMixin, CellMixin, BacktrackMixin):
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
        # self._current_cell: tuple[int, int]

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


class HuntAndKillMazeBuilder(MazeBuilder, CheckStatusMixin, CellMixin, BacktrackMixin):
    """
    The Hunt and Kill algorithm is a modification of the Backtracker algorithm.

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

    # prims_maze = RandomisedPrimsMazeBuilder(int(w), int(h))
    # prims_maze.build_maze()
    #
    # thick_walls = MazePresenterThickStyle(prims_maze.build_steps, 10)
    # thin_walls = MazePresenterThinStyle(prims_maze.build_steps, 10)
    # thick_walls.maze_to_img().show()
    # thin_walls.maze_to_img().show()
    # thin_walls.maze_to_animation('maze_prims3', 120)
    #
    # kruskals_maze = RandomisedKruskalsMazeBuilder(int(w), int(h))
    # kruskals_maze.build_maze()
    #
    # thin_walls = MazePresenterThinStyle(kruskals_maze.build_steps, 10)
    # thin_walls.maze_to_img().show()
    # thin_walls.maze_to_animation('maze_kruskals3', 120)

    # backtracker_maze = BacktrackerMazeBuilder(int(w), int(h))
    # backtracker_maze.build_maze()
    #
    # thin_walls = MazePresenterThinStyle(backtracker_maze.build_steps, 10)
    # thin_walls.maze_to_img().show()
    # thin_walls.maze_to_animation('maze_backtracker4', 120)
    #
    # huntkill_maze = HuntAndKillMazeBuilder(int(w), int(h))
    # huntkill_maze.build_maze()
    #
    # thin_walls = MazePresenterThinStyle(huntkill_maze.build_steps, 10)
    # thin_walls.maze_to_img().show()
    # thin_walls.maze_to_animation('maze_huntkill4', 120)

    bacteria_maze = BacterialGrowthMazeBuilder(int(w), int(h))
    bacteria_maze.build_maze()

    thin_walls = MazePresenterThinStyle(bacteria_maze.build_steps, 10)
    thin_walls.maze_to_img().show()
    thin_walls.maze_to_animation('maze_bacteria4', 120)
