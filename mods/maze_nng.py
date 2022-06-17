from __future__ import annotations

import math
import random
import statistics
import matplotlib.pyplot as plt
import numpy as np
from abc import ABC, abstractmethod
from typing import Generator
from PIL import Image, ImageDraw
from enum import IntEnum


class Cell:
    """
    Represents a graph node = maze cell.
    Each cell contains a link to its neighbours.
    """
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


class CircCell(Cell):
    """
    A modified graph node = maze cell, used for circular mazes represented in polar coordinates.
    Each cell can has more than one bottom neighbour.
    """
    def __init__(self, row: int, column: int):
        super().__init__(row, column)
        self.bottom = []

    def neighbours(self) -> list[Cell | None]:
        lst = []
        if self.bottom: lst.extend(self.bottom)

        if self.top: lst.append(self.top)
        if self.right: lst.append(self.right)
        if self.left: lst.append(self.left)

        return lst


class Grid:
    """
    The maze, made of numerous cells linked together.
    """
    def __init__(self, width: int, height: int):
        self._width = width
        self._height = height
        self.cells: list[list[Cell]] = []

        self._prepare_grid()
        self._configure_cells()

    def _prepare_grid(self) -> None:
        """Initialises the geometry = rectangular maze of size (a x b)."""
        for row in range(self._height):
            self.cells.append(list())
            for col in range(self._width):
                self.cells[row].append(Cell(row, col))

    def _configure_cells(self) -> None:
        """Links each cell to its neighbours."""
        for row in self.cells:
            for cell in row:
                row_, col_ = cell.row, cell.column

                cell.top = self._create_neighbours(row_ - 1, col_)
                cell.bottom = self._create_neighbours(row_ + 1, col_)
                cell.right = self._create_neighbours(row_, col_ + 1)
                cell.left = self._create_neighbours(row_, col_ - 1)

    def _create_neighbours(self, row, column) -> Cell | None:
        if 0 <= row <= self._width - 1 and 0 <= column <= self._height - 1:
            return self.cells[row][column]
        else:
            return None

    def get_random_cell(self) -> Cell:
        return self.cells[random.randrange(0, self._height)][random.randrange(0, self._width)]

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


class CircGrid(Grid):
    """
    Modified maze representation for circular mazes.

    .. note::
        Top most cell is the central one.
    """
    def __init__(self, height: int):
        super().__init__(1, height)

    def _prepare_grid(self) -> None:
        self.cells.append(list())
        self.cells[0].append(CircCell(0, 0))
        self._size = 1

        for row in range(1, self._height):
            self.cells.append(list())

            previous_count = len(self.cells[row - 1])
            ratio = round((row * 2 * math.pi) / previous_count)
            cell_count = previous_count * ratio

            for col in range(cell_count):
                self.cells[row].append(CircCell(row, col))

            self._size += cell_count

    def _configure_cells(self) -> None:
        for row in self.cells:
            for cell in row:
                row_, col_ = cell.row, cell.column

                if row_ > 0:
                    if col_ != 0:
                        cell.left = self.cells[row_][col_ - 1]
                    else:
                        cell.left = self.cells[row_][-1]

                    if col_ != len(row) - 1:
                        cell.right = self.cells[row_][col_ + 1]
                    else:
                        cell.right = self.cells[row_][0]

                    ratio = len(self.cells[row_]) // len(self.cells[row_ - 1])
                    parent = self.cells[row_ - 1][col_ // ratio]
                    cell.top = parent
                    parent.bottom.append(cell)

    def get_random_cell(self) -> Cell:
        row = random.randrange(0, self._height)
        return self.cells[row][random.randrange(0, len(self.cells[row]))]

    @property
    def size(self) -> int:
        return self._size


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
    """Prim's algo using python list to store the frontier cells. """
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
    """Prim's algo (same as above) using python set to store the frontier cells. """
    def __init__(self, grid: Grid):
        self.grid = grid
        self.frontier_cells: set[Cell] = set()

    def build_maze(self):
        self.frontier_cells.add(self.grid.get_random_cell())

        while self.frontier_cells:
            current_cell = random.choice(list(self.frontier_cells))
            self.frontier_cells.remove(current_cell)

            if neighbours := {cell for cell in current_cell.neighbours() if not cell.has_linked_cells()}:
                self.frontier_cells = self.frontier_cells | neighbours

            if in_cells := [cell for cell in current_cell.neighbours() if cell.has_linked_cells()]:
                in_cell = random.choice(in_cells)
                in_cell.link_to(current_cell)
            else:
                current_cell.link_to(current_cell)


class KruskalsMazeBuilder(MazeBuilder):
    """
    Kruskal's algo storing the tree sets in a list of sets.

    It requires a linear lookup (complexity of O(n) {min}) of the cell in the tree sets
    to identify the sets to be joined.
    """
    def __init__(self, grid: Grid):
        self.grid = grid
        self._tree_sets: list[set[Cell | None]] = [{cell} for cell in self.grid.get_next_cell()]

    def _get_tree_index(self, cell: Cell) -> int:
        for i, set_ in enumerate(self._tree_sets):
            if cell in set_:
                return i

    def build_maze(self):
        # tree_sets: set[set[Cell]] = set()
        walls = [(c_, n_) for c_ in self.grid.get_next_cell() for n_ in (c_.bottom, c_.right) if n_]

        random.shuffle(walls)
        while walls:
            # cell, neighbour = random.choice(walls)
            # walls.remove((cell, neighbour))
            cell, neighbour = walls.pop()

            i1 = self._get_tree_index(cell)
            assert isinstance(i1, int)
            if neighbour not in self._tree_sets[i1]:
                i2 = self._get_tree_index(neighbour)
                cell.link_to(neighbour)
                self._tree_sets[i1] = self._tree_sets[i1] | self._tree_sets[i2]
                self._tree_sets.remove(self._tree_sets[i2])


class Kruskals2MazeBuilder(MazeBuilder):
    """
    Kruskal's algo storing the tree sets in a dictionary with K: Cell and V: set.

    Using a dictionary (hash map) to reduce the complexity of lookup of the sets to be joined to O(1).
    However, it necessitates a refresh of dict values corresponding to the dict keys (cells)
    that are in the new joined set.
    """
    def __init__(self, grid: Grid):
        self.grid = grid

    def build_maze(self):
        tree_sets: dict[Cell, set[Cell]] = {cell: {cell} for cell in self.grid.get_next_cell()}
        walls = [(c_, n_) for c_ in self.grid.get_next_cell() for n_ in (c_.bottom, c_.right) if n_]

        set_count = self.grid.size
        random.shuffle(walls)
        while walls and set_count > 1:
            # cell, neighbour = random.choice(walls)
            # walls.remove((cell, neighbour))
            cell, neighbour = walls.pop()

            if neighbour not in tree_sets[cell]:
                cell.link_to(neighbour)

                tree_sets[neighbour].update(tree_sets[cell])
                temp = tree_sets[neighbour]

                for key in list(temp):
                    tree_sets[key] = temp

                set_count -= 1


class Kruskals3MazeBuilder(MazeBuilder):
    """
    Another variation on Kruskal's algo using a nested class = State of the tree sets.

    Uses two dictionaries:
        - one to store the id of tree set (dict V) the cell (dict K) is part of;
        - another to store the tree set (dict V) corresponding to a set id (dict K).
    """
    class State:
        def __init__(self, grid: Grid):
            self.grid = grid
            self.neighbours = []
            self.set_id_of_cell: dict[Cell, int] = {}
            self.cells_in_set: dict[int, list[Cell]] = {}

            self.initialise()

        def initialise(self) -> None:
            for i, cell in enumerate(self.grid.get_next_cell()):
                # i = len(self.set_of_cell)

                self.set_id_of_cell[cell] = i
                self.cells_in_set[i] = [cell]

                if cell.bottom: self.neighbours.append((cell, cell.bottom))
                if cell.right: self.neighbours.append((cell, cell.right))

        def is_to_be_joined(self, left: Cell, right: Cell) -> bool:
            return self.set_id_of_cell[left] != self.set_id_of_cell[right]

        def join_sets(self, left: Cell, right: Cell) -> None:
            left.link_to(right)

            il = self.set_id_of_cell[left]
            ir = self.set_id_of_cell[right]
            temp = self.cells_in_set[ir]

            for cell in temp:
                self.cells_in_set[il].append(cell)
                self.set_id_of_cell[cell] = il

            del self.cells_in_set[ir]

    def __init__(self, grid: Grid):
        self.grid = grid

    def build_maze(self):
        state = self.State(self.grid)

        neighbours = state.neighbours
        random.shuffle(neighbours)

        while neighbours:
            left, right = neighbours.pop()
            if state.is_to_be_joined(left, right): state.join_sets(left, right)


class EllersMazeBuilder(MazeBuilder):
    """Eller's algo, storing the tree sets in a list. """
    def __init__(self, grid: Grid):
        self.grid = grid

    @staticmethod
    def _get_tree_index(cell: Cell, container: list) -> int:
        assert container is not None

        for i, set_ in enumerate(container):
            if cell in set_:
                return i

    @staticmethod
    def _create_link_to_next_row(tree_set: set) -> set[Cell]:
        tree_set = list(tree_set)
        random.shuffle(tree_set)
        new_set = set()

        for i in range(0, random.randint(1, len(tree_set))):
            cell = tree_set[i]
            new_set.add(cell.bottom)
            cell.link_to(cell.bottom)

        assert len(new_set) > 0
        return new_set

    def build_maze(self):
        tree_sets: list[set[Cell]] = []

        for row in self.grid.get_next_row():
            is_last_row = row == self.grid.cells[-1]

            for cell in row:
                if cell is not None:
                    if not cell.has_linked_cells():
                        tree_sets.append(set())
                        tree_sets[-1].add(cell)
                        cell.link_to(cell)
                        i = -1
                    else:
                        i = self._get_tree_index(cell, tree_sets)
                        assert i is not None

                    is_each_in_same_tree = cell.left and cell.left in tree_sets[i]

                    if cell.left and not is_each_in_same_tree:
                        is_to_be_joined = bool(random.randrange(0, 2)) or is_last_row

                        if is_to_be_joined:
                            i_l = self._get_tree_index(cell.left, tree_sets)
                            cell.link_to(cell.left)
                            tree_sets[i_l] = tree_sets[i_l] | tree_sets[i]
                            tree_sets.remove(tree_sets[i])

            if not is_last_row:
                for tree_set in tree_sets[:]:
                    tree_sets.remove(tree_set)
                    tree_sets.append(self._create_link_to_next_row(tree_set))
                assert len(tree_sets) > 0


class Ellers2MazeBuilder(MazeBuilder):
    """Eller's algo storing the tree sets in a dict. """
    def __init__(self, grid: Grid):
        self.grid = grid

    @staticmethod
    def _create_link_to_next_row(tree_set: set) -> dict[Cell, set[Cell]]:
        tree_set = list(tree_set)
        random.shuffle(tree_set)
        new_set = set()
        set_: dict[Cell, set[Cell]] = {}

        for i in range(0, random.randint(1, len(tree_set))):
            cell = tree_set[i]
            new_set.add(cell.bottom)
            cell.link_to(cell.bottom)

        assert len(new_set) > 0

        for cell in new_set:
            set_[cell] = new_set
        return set_

    def build_maze(self):
        tree_sets: dict[Cell, set[Cell]] = {}

        for row in self.grid.get_next_row():
            is_last_row = row == self.grid.cells[-1]

            for cell in row:
                if cell is not None:
                    if not cell.has_linked_cells():
                        tree_sets[cell] = {cell}
                        cell.link_to(cell)

                    is_each_in_same_tree = cell.left and cell.left in tree_sets[cell]

                    if cell.left and not is_each_in_same_tree:
                        is_to_be_joined = bool(random.randrange(0, 2)) or is_last_row

                        if is_to_be_joined:
                            cell.link_to(cell.left)

                            tree_sets[cell.left].update(tree_sets[cell])
                            temp = tree_sets[cell.left]

                            for key in list(temp):
                                tree_sets[key] = temp

            if not is_last_row:
                old_sets = {tuple(tree_sets[k]) for k in tree_sets}
                tree_sets.clear()

                for tree_set in old_sets:
                    tree_sets.update(self._create_link_to_next_row(tree_set))
                assert len(tree_sets) > 0


class HuntAndKillMazeBuilder(MazeBuilder):
    """
    Hunt and Kill algo (a modification of the Recursive backtracking algo),
    which stores all the unvisited neighbours in a set. When the 'random walk' hits a dead-end
    the algo picks from this set to continue the path carving, without the need to do the backtracking.
    """
    def __init__(self, grid: Grid):
        self.grid = grid

    def build_maze(self) -> None:
        current_cell = self.grid.get_random_cell()
        unvisited_cells: set[Cell] = {current_cell}

        while unvisited_cells:
            if neighbours := {n for n in current_cell.neighbours() if not n.has_linked_cells()}:
                neighbour = random.choice(list(neighbours))
                neighbours.remove(neighbour)
                unvisited_cells = unvisited_cells | neighbours

                current_cell.link_to(neighbour)
                current_cell = neighbour
            else:
                # removes all cells from the unvisited set that became visited after during the 'random walk'
                for cell in list(unvisited_cells):
                    if cell.has_linked_cells():
                        unvisited_cells.remove(cell)

                if unvisited_cells:
                    current_cell = random.choice(list(unvisited_cells))
                    unvisited_cells.remove(current_cell)

                    visited_cells = [v for v in current_cell.neighbours() if v.has_linked_cells()]
                    assert len(visited_cells) > 0
                    current_cell.link_to(random.choice(visited_cells))


class HuntAndKillScanMazeBuilder(MazeBuilder):
    """
    A variation to the Hunt and Kill algo, where the unvisited neighbours are not memorised.
    When the 'random walk' hits a dead-end the whole maze is scanned cell by cell, until an unvisited cell
    adjacent to a visited one is found. The algo then pics this cell as a start point to do another 'random walk'.
    """
    def __init__(self, grid: Grid):
        self.grid = grid

    @staticmethod
    def _choose_neighbour_of(cell: Cell) -> Cell | None:
        neighbours = []
        if cell.left and not cell.left.has_linked_cells(): neighbours.append(cell.left)
        if cell.right and not cell.right.has_linked_cells(): neighbours.append(cell.right)

        if neighbours:
            neighbour = random.choice(neighbours)
            return neighbour
        return None

    def build_maze(self) -> None:
        current_cell = self.grid.get_random_cell()

        while True:
            if neighbours := {n for n in current_cell.neighbours() if not n.has_linked_cells()}:
                neighbour = random.choice(list(neighbours))

                current_cell.link_to(neighbour)
                current_cell = neighbour
            else:
                is_unvisited_found = False
                for cell in self.grid.get_next_cell():
                    if cell.has_linked_cells():
                        if neighbour := self._choose_neighbour_of(cell):
                            cell.link_to(neighbour)
                            current_cell = neighbour
                            is_unvisited_found = True
                            break
                if not is_unvisited_found:
                    break


class HuntAndKillScan2MazeBuilder(MazeBuilder):
    """
    Simple modification of the previous one, which checks only the right and left neighbours of the visited cells.
    This one checks all neighbours of the visited cells during the scan.
    """
    def __init__(self, grid: Grid):
        self.grid = grid

    @staticmethod
    def _choose_neighbour_of(cell: Cell, neighbours: set) -> Cell | None:
        neighbour = random.choice(list(neighbours))
        cell.link_to(neighbour)
        return neighbour

    def build_maze(self) -> None:
        current_cell = self.grid.get_random_cell()

        while True:
            if neighbours := {n for n in current_cell.neighbours() if not n.has_linked_cells()}:
                current_cell = self._choose_neighbour_of(current_cell, neighbours)
            else:
                is_unvisited_found = False
                for cell in self.grid.get_next_cell():
                    if not cell.has_linked_cells():
                        if neighbours := {n for n in cell.neighbours() if n.has_linked_cells()}:
                            current_cell = self._choose_neighbour_of(current_cell, neighbours)
                            is_unvisited_found = True
                            break
                if not is_unvisited_found:
                    break


class Label:
    """Context manager used to simulate a label break. """
    class Break(Exception):
        def __init__(self, ctx):
            self.ctx = ctx

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        return isinstance(exc_value, self.Break) and exc_value.ctx is self

    def break_(self):
        raise self.Break(self)


class HuntAndKillScan3MazeBuilder(MazeBuilder):
    """
    Modification of the above one.
    Uses memoization of uncompleted rows (rows with unvisited cells) to speed up the scan part of the algo.

    .. todo:: fix it, doesn't work all the time - most probably the row gets deleted before it is complete.
    """
    def __init__(self, grid: Grid):
        self.grid = grid

    @staticmethod
    def _choose_neighbour_of(cell: Cell, neighbours: set) -> Cell | None:
        neighbour = random.choice(list(neighbours))
        cell.link_to(neighbour)
        return neighbour

    def build_maze(self) -> None:
        current_cell = self.grid.get_random_cell()
        rows_with_unvisited: list[int] = [row_id for row_id in range(self.grid._height)]

        while True:
            if neighbours := {n for n in current_cell.neighbours() if not n.has_linked_cells()}:
                current_cell = self._choose_neighbour_of(current_cell, neighbours)
            else:
                is_unvisited_found = False
                with Label() as search:
                    for row in rows_with_unvisited[:]:
                        for cell in self.grid.cells[row]:
                            if not cell.has_linked_cells():
                                if neighbours := {n for n in cell.neighbours() if n.has_linked_cells()}:
                                    current_cell = self._choose_neighbour_of(current_cell, neighbours)
                                    is_unvisited_found = True
                                    search.break_()
                        rows_with_unvisited.remove(row)

                if not is_unvisited_found:
                    break


class AldousBroderMazeBuilder(MazeBuilder):
    def __init__(self, grid: Grid):
        self.grid = grid

    def build_maze(self):
        unvisited_count = self.grid.size - 1
        current_cell = self.grid.get_random_cell()

        while unvisited_count > 0:
            neighbour = random.choice(current_cell.neighbours())
            if not neighbour.has_linked_cells():
                current_cell.link_to(neighbour)
                current_cell = neighbour

                unvisited_count -= 1
            else:
                current_cell = neighbour


class WilsonsMazeBuilder(MazeBuilder):
    def __init__(self, grid: Grid):
        self.grid = grid

    @staticmethod
    def _do_random_walk(current_cell: Cell) -> dict:
        walked_path = {}

        while True:
            neighbour = random.choice(current_cell.neighbours())
            walked_path[current_cell] = neighbour
            if not neighbour.has_linked_cells():
                current_cell = neighbour
            else:
                return walked_path

    def build_maze(self):
        cell = self.grid.get_random_cell()
        cell.link_to(cell)
        unvisited_count = self.grid.size - 1

        while unvisited_count > 0:
            while True:
                start_cell = self.grid.get_random_cell()
                if not start_cell.has_linked_cells():
                    break

            path = self._do_random_walk(start_cell)

            current_cell = start_cell
            while path:
                unvisited_count -= 1
                if not path[current_cell].has_linked_cells():
                    current_cell.link_to(path[current_cell])
                    current_cell = path[current_cell]
                else:
                    current_cell.link_to(path[current_cell])
                    break


class RecursiveDivisionMazeBuilder(MazeBuilder):
    class Orientation(IntEnum):
        HORIZONTAL = 0
        VERTICAL = 1

    def __init__(self, grid: Grid):
        self.grid = grid

    @staticmethod
    def _choose_orientation(width, height) -> int:
        if width < height:
            return 0
        elif width > height:
            return 1
        else:
            return random.randint(0, 1)

    def build_maze(self):
        x1 = 0
        y1 = 0
        x2 = self.grid._width - 1
        y2 = self.grid._height - 1
        queue: list[tuple[int, int, int, int]] = [(x1, y1, x2, y2)]

        while queue:
            x1, y1, x2, y2 = queue.pop(0)

            if self._choose_orientation(x2 - x1, y2 - y1) == self.Orientation.HORIZONTAL:
                if y2 - y1 + 1 > 1: bisect_y = random.randrange(y1, y2)
                else: bisect_y = y1
                if x2 - x1 + 1 > 1: col = random.randint(x1, x2)
                else: col = x1

                cell = self.grid.cells[bisect_y][col]
                if cell.bottom: cell.link_to(cell.bottom)

                if x2 - x1 + 1 >= 2 or bisect_y - y1 > 0:
                    queue.append((x1, y1, x2, bisect_y))
                if x2 - x1 + 1 >= 2 or y2 - bisect_y > 1:
                    queue.append((x1, bisect_y + 1, x2, y2))
            else:
                if x2 - x1 + 1 > 1:
                    bisect_x = random.randrange(x1, x2)
                else:
                    bisect_x = x1
                if y2 - y1 + 1 > 1:
                    row = random.randint(y1, y2)
                else:
                    row = y1

                cell = self.grid.cells[row][bisect_x]
                if cell.right: cell.link_to(cell.right)

                if y2 - y1 + 1 >= 2 or bisect_x - x1 > 0:
                    queue.append((x1, y1, bisect_x, y2))
                if y2 - y1 + 1 >= 2 or x2 - bisect_x > 1:
                    queue.append((bisect_x + 1, y1, x2, y2))


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
                    output[-1] = self.replace_char(output[-1],
                                                   corners_ver[corners_ver.index(output[-1][5*(j + 1)]) + 1],
                                                   5*(j + 1))
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
        if type(self.grid) == Grid:
            return self._render_rect_grid()
        elif type(self.grid) == CircGrid:
            return self._render_circ_grid()

    def _render_rect_grid(self):
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
        return img

    def _render_circ_grid(self):
        ANTIALIAS_ = 4
        wall_width = self.wall_ * ANTIALIAS_
        offset = wall_width // 2

        w = h = 2*self._set_size(self.grid._height) * ANTIALIAS_

        img = Image.new("RGB", size=(w, h), color=(220, 220, 220))
        draw = ImageDraw.Draw(img)

        for row in range(1, self.grid._height):
            radius1 = (row + 0.75)*self.size * ANTIALIAS_
            radius0 = radius1 - self.size * ANTIALIAS_
            shape = [(w//2 - radius0, h//2 - radius0), (w//2 + radius0, h//2 + radius0)]

            theta = 360 / len(self.grid.cells[row])

            for col in range(len(self.grid.cells[row])):
                cell = self.grid.cells[row][col]

                if not cell.is_linked_to(cell.top):
                    draw.arc(shape, start=(col * theta), end=((col + 1) * theta), fill=(0, 0, 0), width=wall_width)

                if True:#col == 3: #row == 3 and col == 3:
                    radius_ = (row + 0.75) * self.size * ANTIALIAS_
                    circumf_ = radius_ * 2 * math.pi
                    alpha = (360 * (wall_width / 2)) / circumf_
                    shape_ = [(w // 2 - radius1 + wall_width, h // 2 - radius1 + wall_width),
                              (w // 2 + radius1 - wall_width, h // 2 + radius1 - wall_width)]
                    draw.arc(shape_, start=(col * theta) + alpha, end=((col + 1) * theta) - alpha,
                             fill=(250, 150, 150), width=self.size * ANTIALIAS_ - wall_width)

                if not cell.is_linked_to(cell.left):
                    c = np.cos(math.radians(theta) * col)
                    s = np.sin(math.radians(theta) * col)
                    rot_matrix = np.array(((c, -s), (s, c)))

                    x0, y0 = np.dot(rot_matrix, (radius0 - 2*offset, 0))
                    x1, y1 = np.dot(rot_matrix, (radius1, 0))

                    draw.line((x0 + w//2, y0 + h//2, x1 + w//2, y1 + h//2), fill=(0, 0, 0), width=wall_width)

            if row == self.grid._height - 1:
                shape = [(w // 2 - radius1, h // 2 - radius1), (w // 2 + radius1, h // 2 + radius1)]
                draw.arc(shape, start=0, end=360, fill=(0, 0, 0), width=wall_width)

        img = img.resize((w // ANTIALIAS_, h // ANTIALIAS_), resample=Image.ANTIALIAS)
        img.show()
        return img


def plot_statistics(methods: list[str]) -> None:
    import itertools

    colors = itertools.cycle(['r', 'g', 'b', 'c', 'y', 'm', 'k'])
    markers = itertools.cycle(['--', '-.', ':'])
    # filled_markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
    # ls = ('-', '--', '-.', ':')

    for method in methods:
        #global base_time
        x = []
        y = []

        for i in range(20, 100, 10):
            stmt_code = f'grid = Grid({i}, {i}) \nmethod = {method}MazeBuilder(grid) \nmethod.build_maze()'
            time = timeit.repeat(stmt=stmt_code, setup=f'from __main__ import {method}MazeBuilder, Grid', repeat=2,
                                 number=10)
            x.append(i)
            y.append(statistics.mean(time))# / base_time)

        plt.plot(x, y, label=method, c=next(colors), ls=next(markers))

    plt.legend(loc="upper left")
    plt.show()


if __name__ == "__main__":
    # ========================= Binary Tree algo =========================
    # grid = Grid(10, 10)
    # binary = BinaryTreeMazeBuilder(grid)
    # binary.build_maze()
    #
    # ascii_art = AsciiPresenter(grid)
    # ascii_art.to_string()

    # ==================== Recursive backtracker algo ====================
    # grid = Grid(10, 10)
    # backtracker = RecursiveBacktrackerMazeBuilder(grid)
    # backtracker.build_maze()
    #
    # ascii_art = AsciiPresenter(grid)
    # ascii_art.to_string()

    # ========================= Sidewinder algo ==========================
    # grid = Grid(10, 10)
    # sidewinder = SidewinderMazeBuilder(grid)
    # sidewinder.build_maze()
    #
    # ascii_art = AsciiPresenter(grid)
    # ascii_art.to_string()
    #
    # img = ImagePresenter(grid, wall_thickness=2)
    # img.render()

    # =========================== Prim's algo ============================
    # grid = Grid(10, 10)
    # prims = Prims_oldMazeBuilder(grid)
    # prims.build_maze()
    #
    # ascii_art = AsciiPresenter(grid)
    # ascii_art.to_string()
    #
    # img = ImagePresenter(grid, wall_thickness=2)
    # img.render()

    # ========================== Kruskal's algo ==========================
    # grid = Grid(10, 10)
    # kruskals = KruskalsMazeBuilder(grid)
    # kruskals.build_maze()
    #
    # ascii_art = AsciiPresenter(grid)
    # ascii_art.to_string()
    #
    # img = ImagePresenter(grid, wall_thickness=2)
    # img.render()

    # ====================== Kruskal's algo w/ dict ======================
    # grid = Grid(10, 10)
    # kruskals = Kruskals2MazeBuilder(grid)
    # kruskals.build_maze()
    #
    # img = ImagePresenter(grid, wall_thickness=2)
    # img.render()

    # ===================== Kruskal's algo w/ state ======================
    # grid = Grid(10, 10)
    # kruskals = Kruskals3MazeBuilder(grid)
    # kruskals.build_maze()
    #
    # img = ImagePresenter(grid, wall_thickness=2)
    # img.render()

    # =========================== Eller's algo ===========================
    # grid = Grid(10, 10)
    # ellers = EllersMazeBuilder(grid)
    # ellers.build_maze()
    #
    # ascii_art = AsciiPresenter(grid)
    # ascii_art.to_string()
    #
    # img = ImagePresenter(grid, wall_thickness=2)
    # img.render()

    # ======================== Eller's algo v2.0 =========================
    # grid = Grid(10, 10)
    # ellers = Ellers2MazeBuilder(grid)
    # ellers.build_maze()
    #
    # img = ImagePresenter(grid, wall_thickness=2)
    # img.render()
    #
    # # Hunt & Kill algo
    # grid = Grid(10, 10)
    # huntkill = HuntAndKillMazeBuilder(grid)
    # huntkill.build_maze()
    #
    # # ascii_art = AsciiPresenter(grid)
    # # ascii_art.to_string()
    #
    # img = ImagePresenter(grid, wall_thickness=2)
    # img.render()

    # =================== Hunt & Kill algo - scan mode ===================
    # grid = Grid(10, 10)
    # huntkill = HuntAndKillScanMazeBuilder(grid)
    # huntkill.build_maze()
    #
    # # ascii_art = AsciiPresenter(grid)
    # # ascii_art.to_string()
    #
    # img = ImagePresenter(grid, wall_thickness=2)
    # img.render()
    #
    # # Hunt & Kill algo - scan mode v2.0
    # grid = Grid(10, 10)
    # huntkill = HuntAndKillScan2MazeBuilder(grid)
    # huntkill.build_maze()

    # ================ Hunt & Kill algo - scan mode v3.0 =================
    grid = Grid(10, 10)
    huntkill = HuntAndKillScan3MazeBuilder(grid)
    huntkill.build_maze()

    img = ImagePresenter(grid, wall_thickness=2)
    img.render()

    # ======================== Aldous-Broder algo ========================
    # grid = Grid(10, 10)
    # aldousbroder = AldousBroderMazeBuilder(grid)
    # aldousbroder.build_maze()
    #
    # # ascii_art = AsciiPresenter(grid)
    # # ascii_art.to_string()
    #
    # img = ImagePresenter(grid, wall_thickness=2)
    # img.render()

    # ========================== Wilson's algo ===========================
    # grid = Grid(10, 10)
    # wilsons = WilsonsMazeBuilder(grid)
    # wilsons.build_maze()
    #
    # # ascii_art = AsciiPresenter(grid)
    # # ascii_art.to_string()
    #
    # img = ImagePresenter(grid, wall_thickness=2)
    # img.render()

    # ==================== Recursive Division algo =======================
    # grid = Grid(10, 10)
    # division = RecursiveDivisionMazeBuilder(grid)
    # division.build_maze()
    #
    # img = ImagePresenter(grid, wall_thickness=2)
    # img.render()

    # ============================= Timeit ===============================
    import timeit

    plot_statistics(["BinaryTree",
                     "RecursiveBacktracker",
                     "Sidewinder",
                     "Prims_old",
                     "Prims",
                     "Kruskals",
                     "Kruskals2",
                     "Kruskals3",
                     "Ellers",
                     "Ellers2",
                     "HuntAndKill",
                     "HuntAndKillScan",
                     "HuntAndKillScan2",
                     "HuntAndKillScan3",
                     "AldousBroder",
                     "Wilsons",
                     "RecursiveDivision"])

    # ============================= CircGrid =============================
    # grid = CircGrid(10)
    # img = ImagePresenter(grid, wall_thickness=2)
    # img = img.render()
    # RecursiveBacktrackerMazeBuilder(grid).build_maze()

    # img = ImagePresenter(grid, wall_thickness=2)
    # img = img.render()
