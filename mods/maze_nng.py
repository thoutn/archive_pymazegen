from __future__ import annotations

import random
import statistics
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Generator
from PIL import Image, ImageDraw


class Cell:
    def __init__(self, row: int, column: int):
        self.row = row
        self.column = column

        self.links = {}

        self.top = None
        self.bottom = None
        self.right = None
        self.left = None

    def link_to(self, cell: Cell, bidirect=True) -> None:
        self.links[cell] = True
        if bidirect:
            cell.link_to(self, False)

    def unlink_from(self, cell: Cell, bidirect=True) -> None:
        del self.links[cell]
        if bidirect:
            cell.unlink_from(self, False)

    def has_linked_cells(self):
        return self.links.keys()

    def is_linked_to(self, cell) -> bool:
        if cell in self.links.keys():
            return True
        return False

    def neighbours(self) -> list[Cell | None]:
        lst = []
        if self.top: lst.append(self.top)
        if self.bottom: lst.append(self.bottom)
        if self.right: lst.append(self.right)
        if self.left: lst.append(self.left)
        return lst


class Grid:
    def __init__(self, width: int, height: int):
        self._width = width
        self._height = height
        self.cells: list[list[Cell]] = []

        self._prepare_grid()
        self._configure_cells()

    def _prepare_grid(self) -> None:
        for row in range(self._width):
            self.cells.append(list())
            for col in range(self._height):
                self.cells[row].append(Cell(row, col))

    def _configure_cells(self) -> None:
        for row in self.cells:
            for cell in row:
                row, col = cell.row, cell.column

                cell.top = self._create_neighbours(row - 1, col)
                cell.bottom = self._create_neighbours(row + 1, col)
                cell.right = self._create_neighbours(row, col + 1)
                cell.left = self._create_neighbours(row, col - 1)

    def _create_neighbours(self, row, column) -> Cell | None :
        if 0 <= row <= self._width - 1 and 0 <= column <= self._height - 1:
            return self.cells[row][column]
        else:
            return None

    def get_random_cell(self) -> Cell:
        return self.cells[random.randrange(0, self._width)][random.randrange(0, self._height)]

    @property
    def size(self) -> int:
        return self._width * self._height

    def get_next_row(self) -> Generator[list[Cell], None, None]:
        for row in self.cells:
            yield row

    def get_next_cell(self) -> Generator[Cell | None, None, None]:
        for row in self.cells:
            for cell in row:
                yield cell if cell else None


class MazeBuilder(ABC):
    @abstractmethod
    def build_maze(self):
        pass


class BinaryTreeMazeBuilder(MazeBuilder):
    def __init__(self, grid: Grid):
        self.grid = grid

    @staticmethod
    def _choose_neighbour_of(cell: Cell) -> Cell | None:
        neighbours = []
        if cell.top: neighbours.append(cell.top)
        # if cell.top: neighbours.extend([cell.top]*3)
        if cell.right: neighbours.append(cell.right)
        # if cell.right: neighbours.extend([cell.right]*3)

        if neighbours:
            neighbour = random.choice(neighbours)
            return neighbour
        return None

    def build_maze(self) -> None:
        for cell in self.grid.get_next_cell():
            # neighbour = self._choose_neighbour_of(cell)
            if neighbour := self._choose_neighbour_of(cell):
                cell.link_to(neighbour)


class RecursiveBacktrackerMazeBuilder(MazeBuilder):
    def __init__(self, grid: Grid):
        self.grid = grid
        self.stack: list[Cell] = []

    def build_maze(self) -> None:
        self.stack.append(self.grid.get_random_cell())

        while self.stack:
            current_cell = self.stack[-1]

            if neighbours := [cell for cell in current_cell.neighbours() if not cell.has_linked_cells()]:
                random.shuffle(neighbours)
                neighbour = neighbours.pop()

                self.stack.append(neighbour)
                current_cell.link_to(neighbour)
            else:
                self.stack.pop()


class SidewinderMazeBuilder(MazeBuilder):
    def __init__(self, grid: Grid):
        self.grid = grid

    def build_maze(self) -> None:
        for row in self.grid.get_next_row():
            run = []

            for cell in row:
                run.append(cell)

                is_place_to_close_run = cell.right is None or cell.top is not None and random.randrange(0, 2) == 0

                if is_place_to_close_run:
                    cell_ = random.choice(run)
                    if cell_.top: cell_.link_to(cell_.top)
                    run.clear()
                else:
                    cell.link_to(cell.right)


class Prims_oldMazeBuilder(MazeBuilder):
    def __init__(self, grid: Grid):
        self.grid = grid
        self.frontier_cells: list[Cell] = []

    def build_maze(self):
        self.frontier_cells.append(self.grid.get_random_cell())

        while self.frontier_cells:
            random.shuffle(self.frontier_cells)
            current_cell = self.frontier_cells.pop()

            if neighbours := [cell for cell in current_cell.neighbours() if not cell.has_linked_cells()
                              and cell not in self.frontier_cells]:
                self.frontier_cells.extend(neighbours)

            if in_cells := [cell for cell in current_cell.neighbours() if cell.has_linked_cells()]:
                in_cell = random.choice(in_cells)
                in_cell.link_to(current_cell)
            else:
                current_cell.link_to(current_cell)


class PrimsMazeBuilder(MazeBuilder):
    def __init__(self, grid: Grid):
        self.grid = grid
        self.frontier_cells: set[Cell] = set()

    def build_maze(self):
        self.frontier_cells.add(self.grid.get_random_cell())

        while self.frontier_cells:
            current_cell = random.choice(list(self.frontier_cells))
            self.frontier_cells.remove(current_cell)

            if neighbours := {cell for cell in current_cell.neighbours() if not cell.has_linked_cell()}:
                self.frontier_cells = self.frontier_cells | neighbours

            if in_cells := [cell for cell in current_cell.neighbours() if cell.has_linked_cell()]:
                in_cell = random.choice(in_cells)
                in_cell.link_to(current_cell)
            else:
                current_cell.link_to(current_cell)


class KruskalsMazeBuilder(MazeBuilder):
    def __init__(self, grid: Grid):
        self.grid = grid
        self._tree_sets: list[set[Cell | None]] = [{cell} for cell in self.grid.get_next_cell()]

    def get_tree_index(self, cell: Cell) -> int:
        for i, set_ in enumerate(self._tree_sets):
            if cell in set_:
                return i

    def build_maze(self):
        # tree_sets: set[set[Cell]] = set()
        walls = [(c_, n_) for c_ in self.grid.get_next_cell() for n_ in (c_.bottom, c_.right) if n_]

        while walls:
            cell, neighbour = random.choice(walls)
            walls.remove((cell, neighbour))

            i1 = self.get_tree_index(cell)
            assert isinstance(i1, int)
            if neighbour not in self._tree_sets[i1]:
                i2 = self.get_tree_index(neighbour)
                cell.link_to(neighbour)
                self._tree_sets[i1] = self._tree_sets[i1] | self._tree_sets[i2]
                self._tree_sets.remove(self._tree_sets[i2])


class AsciiPresenter:
    def __init__(self, grid: Grid):
        self.size = grid._width
        self.grid = grid

    @staticmethod
    def replace_char(string_: str, char_: str, index_: int) -> str:
        if index_ == -1:
            return string_[:index_] + char_
        else:
            return string_[:index_] + char_ + string_[index_ + 1:]

    def to_string(self) -> None:
        output = ['┏' + "━━━━┳" * (self.size - 1) + "━━━━┓"]

        wall_ver = cor_tb = '┃'
        wall_hor = "━━━━"
        cor_all = '╋'
        cor_tlr = '┻'
        cor_trb = '┣'
        cor_tr = '┗'
        cor_tlb = '┫'
        cor_tl = '┛'
        cor_lrb = '┳'
        cor_lr = '━'
        cor_t = '╹'
        cor_lb = '┓'
        cor_l = '╸'
        cor_rb = '┏'
        cor_r = '╺'
        cor_b = '╻'
        corners_ver = [cor_all, cor_tlr, cor_trb, cor_tr, cor_tlb, cor_tl, cor_lrb, cor_lr, cor_tb, cor_t, cor_lb, cor_l, cor_rb, cor_r]
        corners_hor = [cor_all, cor_tlb, cor_trb, cor_tb, cor_tr, cor_t, cor_lrb, cor_lb, cor_tlr, cor_tl, cor_rb, cor_b]
        cell_body = "    "

        i = 0
        for row in self.grid.get_next_row():
            top_row = wall_ver

            if i < len(self.grid.cells) - 1:
                bottom_row = cor_trb
            else:
                bottom_row = cor_tr

            for j, cell in enumerate(row):
                if not cell:
                    cell = Cell(-1, -1)

                if cell.is_linked_to(cell.right):
                    right_boundary = ' '
                    output[-1] = self.replace_char(output[-1], corners_ver[corners_ver.index(output[-1][5*(j + 1)]) + 1], 5*(j + 1))
                else:
                    right_boundary = wall_ver

                if cell.is_linked_to(cell.bottom):
                    # bottom_row = self.replace_char(bottom_row, corners_hor[corners_hor.index(bottom_row[-1]) + 1], -1)
                    bottom_row = bottom_row[:-1] + corners_hor[corners_hor.index(bottom_row[-1]) + 1]
                    bottom_boundary = "    "
                    if right_boundary == ' ':
                        corner = cor_rb
                    elif j < len(row) - 1:
                        corner = cor_trb
                    else:
                        corner = wall_ver
                else:
                    bottom_boundary = wall_hor
                    if right_boundary == ' ':
                        if i < len(self.grid.cells) - 1:
                            corner = cor_lrb
                        else:
                            corner = cor_lr
                    else:
                        if i < len(self.grid.cells) - 1 and j == len(row) - 1:
                            corner = cor_tlb
                        elif i < len(self.grid.cells) - 1:
                            corner = cor_all
                        elif j < len(row) - 1:
                            corner = cor_tlr
                        else:
                            corner = cor_tl

                top_row += cell_body + right_boundary
                bottom_row += bottom_boundary + corner

            output.append(top_row)
            output.append(bottom_row)

            i += 1

        print('\n'.join(output))


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
            x1 = cell.column * self.size
            y1 = cell.row * self.size
            x2 = (cell.column + 1) * self.size
            y2 = (cell.row + 1) * self.size

            if self.wall_ % 2 != 0:
                offset1 = self.wall_//2
                offset2 = 0
            else:
                offset1 = self.wall_//2 - 1  # 0-2 1-4 2-6 3-8 4-10
                offset2 = 1

            def draw_line(a, b, c, d):
                draw.line((a, b, c, d), fill=(0, 0, 0), width=self.wall_)

            if not cell.top: draw_line(x1, y1 + offset1, x2 + 2*offset1 + offset2, y1 + offset1)
            if not cell.left: draw_line(x1 + offset1, y1, x1 + offset1, y2 + 2*offset1)

            if not cell.is_linked_to(cell.bottom): draw_line(x1, y2 + offset1, x2 + 2*offset1 + offset2, y2 + offset1)
            if not cell.is_linked_to(cell.right): draw_line(x2 + offset1, y1, x2 + offset1, y2 + 2*offset1)

        img.show()


def plot_statistics(methods: list[str]) -> None:
    for method in methods:
        x = []
        y = []
        for i in range(20, 100, 10):
            stmt_code = f'grid = Grid({i}, {i}) \nmethod = {method}MazeBuilder(grid) \nmethod.build_maze()'
            time = timeit.repeat(stmt=stmt_code, setup=f'from __main__ import {method}MazeBuilder, Grid', repeat=2,
                                 number=10)
            x.append(i)
            y.append(statistics.mean(time))

        plt.plot(x, y, label=method)

    plt.legend(loc="upper left")
    plt.show()


if __name__ == "__main__":
    # # Binary Tree algo
    # grid = Grid(10, 10)
    # binary = BinaryTreeMazeBuilder(grid)
    # binary.build_maze()
    #
    # ascii_art = AsciiPresenter(grid)
    # ascii_art.to_string()
    #
    # # Recursive backtracker algo
    # grid = Grid(10, 10)
    # backtracker = RecursiveBacktrackerMazeBuilder(grid)
    # backtracker.build_maze()
    #
    # ascii_art = AsciiPresenter(grid)
    # ascii_art.to_string()
    #
    # # Sidewinder algo
    # grid = Grid(10, 10)
    # sidewinder = SidewinderMazeBuilder(grid)
    # sidewinder.build_maze()
    #
    # ascii_art = AsciiPresenter(grid)
    # ascii_art.to_string()
    #
    # img = ImagePresenter(grid, wall_thickness=2)
    # img.render()
    #
    # # Prim's algo
    # grid = Grid(10, 10)
    # prims = Prims_oldMazeBuilder(grid)
    # prims.build_maze()
    #
    # ascii_art = AsciiPresenter(grid)
    # ascii_art.to_string()
    #
    # img = ImagePresenter(grid, wall_thickness=2)
    # img.render()
    #
    # # Kruskal's algo
    # grid = Grid(10, 10)
    # kruskals = KruskalsMazeBuilder(grid)
    # kruskals.build_maze()
    #
    # ascii_art = AsciiPresenter(grid)
    # ascii_art.to_string()
    #
    # img = ImagePresenter(grid, wall_thickness=2)
    # img.render()

    # Timeit
    import timeit

    # # 0.13 - 0.13
    # stmt_code = "grid = Grid(20, 20) \nbinary = BinaryTreeMazeBuilder(grid) \nbinary.build_maze()"
    # time = timeit.repeat(stmt=stmt_code, setup="from __main__ import BinaryTreeMazeBuilder, Grid", repeat=5,
    #                      number=100)
    # print(f'Binary Tree algo: {time}')
    #
    # # 0.22 - 0.26
    # stmt_code = "grid = Grid(20, 20) \nbacktracker = RecursiveBacktrackerMazeBuilder(grid) \nbacktracker.build_maze()"
    # time = timeit.repeat(stmt=stmt_code, setup="from __main__ import RecursiveBacktrackerMazeBuilder, Grid", repeat=5,
    #                      number=100)
    # print(f'Recursive backtracker algo: {time}')
    #
    # stmt_code = "grid = Grid(20, 20) \nsidewinder = SidewinderMazeBuilder(grid) \nsidewinder.build_maze()"
    # time = timeit.repeat(stmt=stmt_code, setup="from __main__ import SidewinderMazeBuilder, Grid", repeat=5,
    #                      number=100)
    # print(f'Sidewinder algo: {time}')
    #
    # stmt_code = "grid = Grid(20, 20) \nprims = Prims_oldMazeBuilder(grid) \nprims.build_maze()"
    # time = timeit.repeat(stmt=stmt_code, setup="from __main__ import Prims_oldMazeBuilder, Grid", repeat=5,
    #                      number=100)
    # print(f'Prim\'s algo w/ list: {time}')
    #
    # stmt_code = "grid = Grid(20, 20) \nprims = PrimsMazeBuilder(grid) \nprims.build_maze()"
    # time = timeit.repeat(stmt=stmt_code, setup="from __main__ import PrimsMazeBuilder, Grid", repeat=5,
    #                      number=100)
    # print(f'Prim\'s algo w/ set: {time}')

    # Kruskal's: 0.88 - 0.91 (20x20); 32.6 - 35.1 (50x50)
    stmt_code = "grid = Grid(50, 50) \nkruskals = KruskalsMazeBuilder(grid) \nkruskals.build_maze()"
    time = timeit.repeat(stmt=stmt_code, setup="from __main__ import KruskalsMazeBuilder, Grid", repeat=5,
                         number=100)
    print(f'Kruskal\'s algo w/ set: {time}')

    # plot_statistics(["BinaryTree", "RecursiveBacktracker", "Sidewinder", "Prims_old", "Prims", "Kruskals"])

