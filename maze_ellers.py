import random
from random import shuffle

from maze_mixins import CheckStatusMixin, CellMixin, COLOUR_WHITE2, COLOUR_LRED, COLOUR_RED2
from maze_type import MazeBuilder
from presenter import MazePresenterThickStyle, MazePresenterThinStyle


class EllersMazeBuilder(MazeBuilder, CheckStatusMixin, CellMixin):
    def __init__(self, width: int, height: int):
        super().__init__(width, height)
        self._tree_sets: list[set[tuple[int, int]]] = []

    # to delete
    def _is_marked(self, xx, yy):
        return self._maze[yy][xx] == COLOUR_WHITE2

    # to delete
    def _save_cell(self, xx, yy):
        self._mark_cell(xx, yy)

    def build_maze(self):
        def save_vertical_connections(xx: int, yy: int):
            for y in range(yy + 1, yy + 3):
                self._save_cell(xx, y)
            self._tree_sets[-1].add((xx, yy + 2))

        def create_vertical_connections_to_next_row() -> None:
            # self._in_cells.clear()
            for set_ in self._tree_sets[:]:
                self._tree_sets.remove(set_)

                set_ = list(set_)
                shuffle(set_)
                self._tree_sets.append(set())

                for i in range(0, len(set_), random.randint(1, len(set_))):
                    x, y = set_[i]
                    save_vertical_connections(x, y)

        def initialise_new_sets_at(y: int) -> None:
            for x in range(1, self.width - 1, 2):
                if not self._is_marked(x, y):
                    self._tree_sets.append(set())
                    self._tree_sets[-1].add((x, y))
                    self._save_cell(x, y)

        def is_to_be_joined() -> bool:
            return bool(random.randint(0, 1))

        def is_last_row(row_: int) -> bool:
            return row_ == self.height - 2

        def mark_cells_at_wall(w, c1, c2):
            self._maze[c1[1]][c1[0]] = COLOUR_LRED
            self._maze[c2[1]][c2[0]] = COLOUR_LRED
            self._maze[w[1]][w[0]] = COLOUR_RED2

        def randomly_join_adjacent_cells_at(y: int):
            for x in range(3, self.width - 1, 2):
                if not is_each_cell_in_same_tree((x, y), (x - 2, y)):
                    if is_to_be_joined() or is_last_row(y):
                        mark_cells_at_wall((x - 1, y), (x, y), (x - 2, y))
                        self._save_next_anim_frame()

                        join_cells((x - 2, y), (x, y))
                        self._save_next_anim_frame()

        def is_each_cell_in_same_tree(c1, c2) -> bool:
            for set_ in self._tree_sets:
                if c1 in set_:
                    if c2 in set_:
                        return True
                    else:
                        return False

        def get_set_with_cell(cell_: tuple[int, int]) -> int:
            for i, set_ in enumerate(self._tree_sets):
                if cell_ in set_:
                    return i

        def join_tree_sets(c1, c2):
            i1 = get_set_with_cell(c1)
            i2 = get_set_with_cell(c2)

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

        def carve_passage():
            for row in range(1, self.height - 1, 2):
                initialise_new_sets_at(row)
                self._save_next_anim_frame()

                randomly_join_adjacent_cells_at(row)

                if row != self.height - 2:
                    create_vertical_connections_to_next_row()
                    self._save_next_anim_frame()

        self._save_next_anim_frame()
        carve_passage()

        self._add_maze_start_point()
        self._add_maze_end_point()


if __name__ == '__main__':
    size = input("Enter maze size as \"width\"x\"height\": ")
    w, h = size.split('x')

    ellers_maze = EllersMazeBuilder(int(w), int(h))
    ellers_maze.build_maze()

    thick_walls = MazePresenterThickStyle(ellers_maze.build_steps, 10)
    thick_walls.maze_to_img().show()

    thin_walls = MazePresenterThinStyle(ellers_maze.build_steps)
    thin_walls.maze_to_img().show()

    thin_walls.maze_to_animation('maze_ellers', 120)

    # import timeit
    # stmt_code = "ellers_maze = EllersMazeBuilder(50, 50) \nellers_maze.build_maze()"
    # time = timeit.repeat(stmt=stmt_code, setup="from __main__ import EllersMazeBuilder", repeat=5,
    #                      number=10)
    # print(time)
