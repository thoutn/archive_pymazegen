import math
import random
import statistics
import matplotlib.pyplot as plt
import numpy as np
from abc import ABC, abstractmethod
from typing import Generator
from enum import IntEnum


cdef class Cell:
    cdef public int row, column
    cdef dict links
    cdef public Cell top, bottom, right, left

    def __init__(self, int row, int column):
        self.row = row
        self.column = column

        self.links = {}

        self.top = None
        self.bottom = None
        self.right = None
        self.left = None

    cpdef void link_to(self, Cell cell, bint bidirect=True):
        self.links[cell] = True
        if bidirect:
            cell.link_to(self, False)

    cpdef void unlink_from(self, Cell cell, bint bidirect=True):
        del self.links[cell]
        if bidirect:
            cell.unlink_from(self, False)

    cpdef has_linked_cells(self):
        return self.links.keys()

    cpdef bint is_linked_to(self, Cell cell):
        if cell in self.links.keys():
            return True
        return False

    cpdef list neighbours(self):
        cdef list lst = []
        if self.top: lst.append(self.top)
        if self.bottom: lst.append(self.bottom)
        if self.right: lst.append(self.right)
        if self.left: lst.append(self.left)
        return lst


cdef class Grid:
    cdef public int width, height
    cdef public list cells

    def __init__(self, int width, int height):
        self.width = width
        self.height = height
        self.cells = []

        self._prepare_grid()
        self._configure_cells()

    cdef void _prepare_grid(self):
        cdef int row, col
        for row in range(self.height):
            self.cells.append(list())
            for col in range(self.width):
                self.cells[row].append(Cell(row, col))

    cdef void _configure_cells(self):
        cdef int row_, col_
        cdef Cell cell
        # cdef int row, col
        for row_ in range(self.height):
            for col_ in range(self.width):
                cell = self.cells[row_][col_]
                # row_, col_ = cell.row, cell.column

                cell.top = self._create_neighbours(row_ - 1, col_)
                cell.bottom = self._create_neighbours(row_ + 1, col_)
                cell.right = self._create_neighbours(row_, col_ + 1)
                cell.left = self._create_neighbours(row_, col_ - 1)

    cdef Cell _create_neighbours(self, int row, int column):
        if 0 <= row <= self.width - 1 and 0 <= column <= self.height - 1:
            return self.cells[row][column]
        else:
            return None

    cpdef Cell get_random_cell(self):
        return self.cells[random.randrange(0, self.height)][random.randrange(0, self.width)]

    cpdef int size(self):
        return self.width * self.height

    # def get_next_row(self) -> Generator[list[Cell], None, None]:
    #     for row in self.cells:
    #         yield row
    #
    # def get_next_cell(self) -> Generator[Cell | None, None, None]:
    #     for row in self.cells:
    #         for cell in row:
    #             yield cell if cell else None


cdef class BinaryTreeMazeBuilder():
    cdef Grid grid
    def __init__(self, Grid grid):
        self.grid = grid

    cdef Cell _choose_neighbour_of(self, Cell cell):
        cdef list neighbours = []
        if cell.top: neighbours.append(cell.top)
        # if cell.top: neighbours.extend([cell.top]*3)
        if cell.right: neighbours.append(cell.right)
        # if cell.right: neighbours.extend([cell.right]*3)

        cdef Cell neighbour
        if neighbours:
            neighbour = random.choice(neighbours)
            return neighbour
        return None

    cpdef void build_maze(self):
        cdef Cell cell, neighbour
        cdef int row, col

        for row in range(self.grid.height):
            for col in range(self.grid.width):
                cell = self.grid.cells[row][col]
                neighbour = self._choose_neighbour_of(cell)
                if neighbour:
                    cell.link_to(neighbour)


# class RecursiveBacktrackerMazeBuilder(MazeBuilder):
#     def __init__(self, grid: Grid):
#         self.grid = grid
#         self.stack: list[Cell] = []
#
#     def build_maze(self) -> None:
#         self.stack.append(self.grid.get_random_cell())
#
#         while self.stack:
#             current_cell = self.stack[-1]
#
#             if neighbours := [cell for cell in current_cell.neighbours() if not cell.has_linked_cells()]:
#                 random.shuffle(neighbours)
#                 neighbour = neighbours.pop()
#
#                 self.stack.append(neighbour)
#                 current_cell.link_to(neighbour)
#             else:
#                 self.stack.pop()


cdef class State:
    cdef Grid grid
    cdef list neighbours
    cdef dict set_id_of_cell, cells_in_set

    def __init__(self, Grid grid):
        self.grid = grid
        self.neighbours = []
        self.set_id_of_cell = {}
        self.cells_in_set = {}

        self.initialise()

    cdef void initialise(self):
        cdef int i, row, col
        cdef Cell cell

        for row in range(self.grid.height):
            for col in range(self.grid.width):
                cell = self.grid.cells[row][col]
                i += 1
                # i = len(self.set_of_cell)

                self.set_id_of_cell[cell] = i
                self.cells_in_set[i] = [cell]

                if cell.bottom: self.neighbours.append((cell, cell.bottom))
                if cell.right: self.neighbours.append((cell, cell.right))

    cdef bint is_to_be_joined(self, Cell left, Cell right):
        return self.set_id_of_cell[left] != self.set_id_of_cell[right]

    cdef void join_sets(self, Cell left, Cell right):
        left.link_to(right)

        cdef il, ir
        cdef list temp
        il = self.set_id_of_cell[left]
        ir = self.set_id_of_cell[right]
        temp = self.cells_in_set[ir]

        cdef Cell cell
        for cell in temp:
            self.cells_in_set[il].append(cell)
            self.set_id_of_cell[cell] = il

        del self.cells_in_set[ir]


cdef class Kruskals3MazeBuilder():
    cdef Grid grid
    def __init__(self, Grid grid):
        self.grid = grid

    cpdef build_maze(self):
        state = State(self.grid)

        cdef list neighbours
        neighbours = state.neighbours
        random.shuffle(neighbours)

        cdef Cell left, right
        while neighbours:
            left, right = neighbours.pop()
            if state.is_to_be_joined(left, right): state.join_sets(left, right)
