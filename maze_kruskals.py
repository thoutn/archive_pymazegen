from random import shuffle

from maze_mixins import CheckStatusMixin, CellMixin, COLOUR_RED, COLOUR_RED2, COLOUR_YELLOW
from maze_type import MazeBuilder
from presenter import MazePresenterThickStyle, MazePresenterThinStyle


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

        def is_each_cell_in_same_tree(c1, c2) -> bool:
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

                if not is_each_cell_in_same_tree(cell_1, cell_2):
                    join_cells(cell_1, cell_2)
                else:
                    solidify_wall(cell_1, cell_2)

        carve_passage()

        self._add_maze_start_point()
        self._add_maze_end_point()


if __name__ == '__main__':
    # size = input("Enter maze size as \"width\"x\"height\": ")
    # w, h = size.split('x')
    #
    # kruskals_maze = RandomisedKruskalsMazeBuilder(int(w), int(h))
    # kruskals_maze.build_maze()
    #
    # thick_walls = MazePresenterThickStyle(kruskals_maze.build_steps, 10)
    # thick_walls.maze_to_img().show()
    #
    # thin_walls = MazePresenterThinStyle(kruskals_maze.build_steps)
    # thin_walls.maze_to_img().show()
    #
    # thin_walls.maze_to_animation('maze_kruskals', 120)

    import timeit
    stmt_code = "kruskals_maze = RandomisedKruskalsMazeBuilder(50, 50) \nkruskals_maze.build_maze()"
    time = timeit.repeat(stmt=stmt_code, setup="from __main__ import RandomisedKruskalsMazeBuilder", repeat=5,
                         number=10)
    print(time)
