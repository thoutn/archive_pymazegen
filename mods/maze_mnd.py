from __future__ import annotations

from enum import Enum
import random
from abc import ABC, abstractmethod
from typing import Generator
from PIL import Image, ImageDraw


class Dir(Enum):
    TOP = 1
    BOTTOM = 2
    RIGHT = 3
    LEFT = 4


class Grid:
    """
    Combines the node and matrix representation.
    Stores only the cells in a matrix, with a dictionary holding the links to the cell's neighbours.
    Pretty much same as 'maze_nng' variant, except it doesn't need the creation and multiple instantiation
    of the 'Cell' class.
    """
    def __init__(self, width: int, height: int) -> None:
        self._width = width
        self._height = height
        self.cells: list[list[dict]] = []

        self._prepare_grid()
        self._configure_cells()

    def _prepare_grid(self) -> None:
        for row in range(self._width):
            self.cells.append(list())
            for col in range(self._height):
                self.cells[row].append({})

    def _configure_cells(self) -> None:
        for row in range(self._height):
            for col in range(self._width):
                self.cells[row][col][(row - 1, col)] = self._create_neighbours(row - 1, col)
                self.cells[row][col][(row + 1, col)] = self._create_neighbours(row + 1, col)
                self.cells[row][col][(row, col + 1)] = self._create_neighbours(row, col + 1)
                self.cells[row][col][(row, col - 1)] = self._create_neighbours(row, col - 1)

    def _create_neighbours(self, row, col) -> bool | None:
        if 0 <= row <= self._width - 1 and 0 <= col <= self._height - 1:
            return False
        else:
            return None

    def link_cells(self, cell_a: tuple, cell_b: tuple, bidirect: bool = True) -> None:
        row, col = cell_a
        self.cells[row][col][cell_b] = True
        if bidirect:
            self.link_cells(cell_b, cell_a, False)

    def is_link_between(self, cell: tuple, n_: Dir) -> bool:
        row, col = cell
        if n_ == Dir.TOP:
            return self.cells[row][col][(row - 1, col)]
        elif n_ == Dir.BOTTOM:
            return self.cells[row][col][(row + 1, col)]
        elif n_ == Dir.RIGHT:
            return self.cells[row][col][(row, col + 1)]
        elif n_ == Dir.LEFT:
            return self.cells[row][col][(row, col - 1)]
        # if cell_b in self.cells[row][col].keys():
        #     return True
        # return False

    def has_neighbour(self, cell: tuple, n_: Dir) -> bool:
        row, col = cell
        if n_ == Dir.TOP and self.cells[row][col][(row - 1, col)] is not None: return True
        if n_ == Dir.BOTTOM and self.cells[row][col][(row + 1, col)] is not None: return True
        if n_ == Dir.RIGHT and self.cells[row][col][(row, col + 1)] is not None: return True
        if n_ == Dir.LEFT and self.cells[row][col][(row, col - 1)] is not None: return True
        return False

    def get_random_cell(self) -> tuple[int, int]:
        # return self.cells[random.randrange(0, self._height)][random.randrange(0, self._width)]
        return random.randrange(0, self._height), random.randrange(0, self._width)

    def get_next_row(self) -> Generator[int, None, None]:
        for row in range(self._height):
            yield row

    def get_next_cell(self) -> Generator[tuple[int, int], None, None]:
        for row in range(self._height):
            for col in range(self._width):
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
        if self.grid.has_neighbour(cell, Dir.TOP): neighbours.append((row - 1, col))
        if self.grid.has_neighbour(cell, Dir.RIGHT): neighbours.append((row, col + 1))

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
        self.stack: list[tuple[int, int]] = []

    def _is_inside_grid(self, row, col, cell):
        return self.grid.cells[row][col][cell] is not None

    def _has_linked_cell(self, cell):
        row, col = cell
        return True in self.grid.cells[row][col].values()

    def build_maze(self) -> None:
        self.stack.append(self.grid.get_random_cell())

        while self.stack:
            row, col = self.stack[-1]

            if neighbours := [cell for cell in self.grid.cells[row][col].keys()
                              if self._is_inside_grid(row, col, cell) and not self._has_linked_cell(cell)]:
                neighbour = random.choice(neighbours)

                self.stack.append(neighbour)
                self.grid.link_cells((row, col), neighbour)
            else:
                self.stack.pop()


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
            x1 = col * self.size
            y1 = row * self.size
            x2 = (col + 1) * self.size
            y2 = (row + 1) * self.size

            if self.wall_ % 2 != 0:
                offset1 = self.wall_//2
                offset2 = 0
            else:
                offset1 = self.wall_//2 - 1  # 0-2 1-4 2-6 3-8 4-10
                offset2 = 1

            def draw_line(a, b, c, d):
                draw.line((a, b, c, d), fill=(0, 0, 0), width=self.wall_)

            if not self.grid.has_neighbour(cell, Dir.TOP):
                draw_line(x1, y1 + offset1, x2 + 2*offset1 + offset2, y1 + offset1)
            if not self.grid.has_neighbour(cell, Dir.LEFT):
                draw_line(x1 + offset1, y1, x1 + offset1, y2 + 2*offset1)

            if not self.grid.is_link_between(cell, Dir.BOTTOM):
                draw_line(x1, y2 + offset1, x2 + 2*offset1 + offset2, y2 + offset1)
            if not self.grid.is_link_between(cell, Dir.RIGHT):
                draw_line(x2 + offset1, y1, x2 + offset1, y2 + 2*offset1)

        img.show()


if __name__ == '__main__':
    # Binary Tree algo
    grid = Grid(10, 10)
    binary = BinaryTreeMazeBuilder(grid)
    binary.build_maze()

    img = ImagePresenter(grid, wall_thickness=2)
    img.render()

    # Recursive backtracker algo
    grid = Grid(10, 10)
    backtracker = RecursiveBacktrackerMazeBuilder(grid)
    backtracker.build_maze()

    img = ImagePresenter(grid, wall_thickness=2)
    img.render()

    # Timeit
    import timeit

    # 0.16 - 0.23
    stmt_code = "grid = Grid(20, 20) \nbinary = BinaryTreeMazeBuilder(grid) \nbinary.build_maze()"
    time = timeit.repeat(stmt=stmt_code, setup="from __main__ import BinaryTreeMazeBuilder, Grid", repeat=5,
                         number=100)
    print(f'Binary Tree algo: {time}')

    # 0.29 - 0.41
    stmt_code = "grid = Grid(20, 20) \nbacktracker = RecursiveBacktrackerMazeBuilder(grid) \nbacktracker.build_maze()"
    time = timeit.repeat(stmt=stmt_code, setup="from __main__ import RecursiveBacktrackerMazeBuilder, Grid", repeat=5,
                         number=100)
    print(f'Recursive backtracker algo: {time}')
