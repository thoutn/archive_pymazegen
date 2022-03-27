from __future__ import annotations

import random
from typing import Generator
from abc import ABC, abstractmethod
from PIL import Image, ImageDraw


class Grid:
    def __init__(self, width: int, height: int):
        self._width = width
        self._height = height
        self.cells: list[list[int | None]] = [[0] * (2*self._width + 1) for row in range(2*self._height + 1)]

        self._configure_cells()

    def _configure_cells(self) -> None:
        for row in range(len(self.cells)):
            if row == 0 or row == len(self.cells) - 1:
                for col in range(len(self.cells[row])):
                    self.cells[row][col] = None
            else:
                self.cells[row][0] = None
                self.cells[row][len(self.cells[row]) - 1] = None

    def link_cells(self, cell_a: tuple, cell_b: tuple) -> None:
        row_a, col_a = cell_a
        row_b, col_b = cell_b
        self.cells[row_a + (row_b - row_a)//2][col_a + (col_b - col_a)//2] = 1
        self.cells[row_a][col_a] = 1
        self.cells[row_b][col_b] = 1

    def unlink_cells(self, cell_a: tuple, cell_b: tuple) -> None:
        row_a, col_a = cell_a
        row_b, col_b = cell_b
        self.cells[row_a + (row_b - row_a)//2][col_a + (col_b - col_a)//2] = 0

    def has_linked_cell(self, cell: tuple) -> bool:
        row, col = cell
        if self.cells[row - 1][col]: return True
        if self.cells[row + 1][col]: return True
        if self.cells[row][col + 1]: return True
        if self.cells[row][col - 1]: return True
        return False

    def is_link_between(self, cell_a: tuple, cell_b: tuple) -> bool:
        row_a, col_a = cell_a
        row_b, col_b = cell_b
        if self.cells[row_a + (row_b - row_a)//2][col_a + (col_b - col_a)//2] == 1:
            return True
        return False

    def get_neighbours_of(self, cell: tuple) -> list[tuple[int, int] | None]:
        row, col = cell
        lst = []
        if self.cells[row - 1][col] is not None: lst.append((row - 2, col))
        if self.cells[row + 1][col] is not None: lst.append((row + 2, col))
        if self.cells[row][col + 1] is not None: lst.append((row, col + 2))
        if self.cells[row][col - 1] is not None: lst.append((row, col - 2))
        return lst

    def get_random_cell(self) -> tuple[int, int]:
        width = 2*self._width + 1
        height = 2*self._height + 1
        return random.randrange(1, height, 2), random.randrange(1, width, 2)

    def get_next_row(self) -> Generator[int, None, None]:
        for row in range(1, len(self.cells), 2):
            yield row

    def get_next_cell(self) -> Generator[tuple[int, int], None, None]:
        for row in range(1, len(self.cells), 2):
            for col in range(1, len(self.cells[row]), 2):
                yield row, col


class MazeBuilder(ABC):
    @abstractmethod
    def build_maze(self):
        pass


class BinaryTreeMazeBuilder(MazeBuilder):
    def __init__(self, grid: Grid):
        self.grid = grid

    def _choose_neighbour_of(self, cell: tuple) -> tuple | None:
        row, col = cell
        neighbours = []

        has_top_neighbour = self.grid.cells[row -1][col] is not None
        has_right_neighbour = self.grid.cells[row][col + 1] is not None

        if has_top_neighbour: neighbours.append((row - 2, col))
        if has_right_neighbour: neighbours.append((row, col + 2))

        if neighbours:
            return random.choice(neighbours)
        return None

    def build_maze(self) -> None:
        for cell in self.grid.get_next_cell():
            if neighbour := self._choose_neighbour_of(cell):
                self.grid.link_cells(cell, neighbour)


class RecursiveBacktrackerMazeBuilder(MazeBuilder):
    def __init__(self, grid: Grid):
        self.grid = grid

    def build_maze(self) -> None:
        stack: list[tuple[int, int]] = [self.grid.get_random_cell()]

        while stack:
            current_cell = stack[-1]

            if neighbours := [n for n in self.grid.get_neighbours_of(current_cell)
                              if not self.grid.has_linked_cell(n)]:
                neighbour = random.choice(neighbours)

                stack.append(neighbour)
                self.grid.link_cells(current_cell, neighbour)
            else:
                stack.pop()


class SidewinderMazeBuilder(MazeBuilder):
    def __init__(self, grid: Grid):
        self.grid = grid

    def build_maze(self) -> None:
        for row in self.grid.get_next_row():
            run = []

            for col in range(1, len(self.grid.cells[row]), 2):
                current_cell = (row, col)
                run.append(current_cell)

                has_top_neighbour = self.grid.cells[row - 1][col] is not None
                has_right_neighbour = self.grid.cells[row][col + 1] is not None
                is_place_to_close_run = not has_right_neighbour or has_top_neighbour and random.randrange(0, 2) == 0

                if is_place_to_close_run:
                    cell_ = random.choice(run)
                    top_neighbour = (cell_[0] - 2, cell_[1])
                    if has_top_neighbour: self.grid.link_cells(cell_, top_neighbour)
                    run.clear()
                else:
                    right_neighbour = (row, col + 2)
                    self.grid.link_cells(current_cell, right_neighbour)


class PrimsMazeBuilder(MazeBuilder):
    def __init__(self, grid: Grid):
        self.grid = grid

    def build_maze(self):
        frontier_cells: set[tuple[int, int]] = set()
        frontier_cells.add(self.grid.get_random_cell())

        while frontier_cells:
            current_cell = random.choice(list(frontier_cells))
            frontier_cells.remove(current_cell)

            if neighbours := {n for n in self.grid.get_neighbours_of(current_cell)
                              if not self.grid.has_linked_cell(n)}:
                frontier_cells = frontier_cells | neighbours

            if in_cells := [n for n in self.grid.get_neighbours_of(current_cell)
                            if self.grid.has_linked_cell(n) or self.grid.cells[current_cell[0]][current_cell[1]] == 1]:
                in_cell = random.choice(in_cells)
                self.grid.link_cells(current_cell, in_cell)
            else:
                self.grid.cells[current_cell[0]][current_cell[1]] = 1


class KruskalsMazeBuilder(MazeBuilder):
    def __init__(self, grid: Grid):
        self.grid = grid

    @staticmethod
    def get_tree_index(cell: tuple, container: list) -> int:
        for i, set_ in enumerate(container):
            if cell in set_:
                return i

    def _get_neighbours_of(self, cell: tuple) -> list[tuple[int, int] | None]:
        row, col = cell
        lst = []
        if self.grid.cells[row + 1][col] is not None: lst.append((row + 2, col))
        if self.grid.cells[row][col + 1] is not None: lst.append((row, col + 2))
        return lst

    def build_maze(self):
        tree_sets: list[set[tuple[int, int]]] = [{cell} for cell in self.grid.get_next_cell()]
        walls = [(c_, n_) for c_ in self.grid.get_next_cell() for n_ in self._get_neighbours_of(c_)]

        while walls:
            cell, neighbour = random.choice(walls)
            walls.remove((cell, neighbour))

            i1 = self.get_tree_index(cell, tree_sets)
            assert isinstance(i1, int)
            if neighbour not in tree_sets[i1]:
                i2 = self.get_tree_index(neighbour, tree_sets)
                tree_sets[i1] = tree_sets[i1] | tree_sets[i2]
                tree_sets.remove(tree_sets[i2])

                self.grid.link_cells(cell, neighbour)


class ImagePresenter:
    def __init__(self, grid: Grid, cell_size: int = 20, wall_thickness: int = 2):
        self.grid = grid
        self.cell_ = cell_size
        self.wall_ = wall_thickness
        self.size = (self.cell_ + self.wall_)

    def _set_size(self, size: int) -> int:
        return size * self.cell_ + (size + 1) * self.wall_

    def render(self):
        w = self._set_size(self.grid._width)
        h = self._set_size(self.grid._height)

        img = Image.new("RGB", size=(w, h), color=(220, 220, 220))
        draw = ImageDraw.Draw(img)

        for cell in self.grid.get_next_cell():
            row, col = cell
            x1 = col//2 * self.size
            y1 = row//2 * self.size
            x2 = (col//2 + 1) * self.size
            y2 = (row//2 + 1) * self.size

            if self.wall_ % 2 != 0:
                offset1 = self.wall_//2
                offset2 = 0
            else:
                offset1 = self.wall_//2 - 1  # 0-2 1-4 2-6 3-8 4-10
                offset2 = 1

            def draw_line(a, b, c, d):
                draw.line((a, b, c, d), fill=(0, 0, 0), width=self.wall_)

            if not self.grid.cells[row - 1][col]: # TOP
                draw_line(x1, y1 + offset1, x2 + 2*offset1 + offset2, y1 + offset1)
            if not self.grid.cells[row][col - 1]: # LEFT
                draw_line(x1 + offset1, y1, x1 + offset1, y2 + 2*offset1)

            if not self.grid.is_link_between(cell, (row + 2, col)): # BOTTOM
                draw_line(x1, y2 + offset1, x2 + 2*offset1 + offset2, y2 + offset1)
            if not self.grid.is_link_between(cell, (row, col + 2)): # RIGHT
                draw_line(x2 + offset1, y1, x2 + offset1, y2 + 2*offset1)

        img.show()


if __name__ == '__main__':
    # # Binary Tree algo
    # grid = Grid(10, 10)
    # binary = BinaryTreeMazeBuilder(grid)
    # binary.build_maze()
    #
    # img = ImagePresenter(grid, wall_thickness=2)
    # img.render()
    #
    # # Recursive backtracker algo
    # grid = Grid(10, 10)
    # backtracker = RecursiveBacktrackerMazeBuilder(grid)
    # backtracker.build_maze()
    #
    # img = ImagePresenter(grid, wall_thickness=2)
    # img.render()
    #
    # # Sidewinder algo
    # grid = Grid(10, 10)
    # sidewinder = SidewinderMazeBuilder(grid)
    # sidewinder.build_maze()
    #
    # img = ImagePresenter(grid, wall_thickness=2)
    # img.render()
    #
    # # Prim's algo
    # grid = Grid(10, 10)
    # prims = PrimsMazeBuilder(grid)
    # prims.build_maze()
    #
    # img = ImagePresenter(grid, wall_thickness=2)
    # img.render()
    #
    # # Kruskal's algo
    # grid = Grid(10, 10)
    # kruskals = KruskalsMazeBuilder(grid)
    # kruskals.build_maze()
    #
    # img = ImagePresenter(grid, wall_thickness=2)
    # img.render()

    # Timeit
    import timeit

    # # BinaryTree: 0.058 - 0.095
    # stmt_code = "grid = Grid(20, 20) \nbinary = BinaryTreeMazeBuilder(grid) \nbinary.build_maze()"
    # time = timeit.repeat(stmt=stmt_code, setup="from __main__ import BinaryTreeMazeBuilder, Grid", repeat=5,
    #                      number=100)
    # print(f'Binary Tree algo: {time}')
    #
    # # Backtracker: 0.21 - 0.34
    # stmt_code = "grid = Grid(20, 20) \nbacktracker = RecursiveBacktrackerMazeBuilder(grid) \nbacktracker.build_maze()"
    # time = timeit.repeat(stmt=stmt_code, setup="from __main__ import RecursiveBacktrackerMazeBuilder, Grid", repeat=5,
    #                      number=100)
    # print(f'Recursive backtracker algo: {time}')
    #
    # # Sidewinder: 0.068 -0.11
    # stmt_code = "grid = Grid(20, 20) \nsidewinder = SidewinderMazeBuilder(grid) \nsidewinder.build_maze()"
    # time = timeit.repeat(stmt=stmt_code, setup="from __main__ import SidewinderMazeBuilder, Grid", repeat=5,
    #                      number=100)
    # print(f'Sidewinder algo: {time}')
    #
    # # Prim's: 0.326 - 0.48
    # stmt_code = "grid = Grid(20, 20) \nprims = PrimsMazeBuilder(grid) \nprims.build_maze()"
    # time = timeit.repeat(stmt=stmt_code, setup="from __main__ import PrimsMazeBuilder, Grid", repeat=5,
    #                      number=100)
    # print(f'Prim\'s algo w/ set: {time}')

    # Kruskal's: 1.00 - 1.5 (20x20); 39.2 - 43.1 (50x50)
    stmt_code = "grid = Grid(50, 50) \nkruskals = KruskalsMazeBuilder(grid) \nkruskals.build_maze()"
    time = timeit.repeat(stmt=stmt_code, setup="from __main__ import KruskalsMazeBuilder, Grid", repeat=5,
                         number=100)
    print(f'Kruskal\'s algo w/ set: {time}')
