import numpy as np
import numpy.typing as npt
from PIL import Image
from pprint import pprint
from random import randrange, shuffle
import copy
from enum import IntEnum


CELL_SIZE = 20
COLOUR_BLACK = [0, 0, 0]
COLOUR_WHITE = [255, 255, 255]
COLOUR_WHITE2 = [240, 240, 240]
COLOUR_YELLOW = [255, 211, 67]
COLOUR_YELLOW2 = [204, 156, 0]
COLOUR_GREEN = [202, 224, 12]
COLOUR_GREEN2 = [160, 178, 9]
COLOUR_GRAY = [40, 40, 40]


class Cell(IntEnum):
    START = 0
    END = 1


def resize_maze(maze_: list, multiple: int = CELL_SIZE) -> np.ndarray:
    def resize_rows(list_: np.ndarray) -> np.ndarray:
        enlarged = [[row] * multiple for row in list_]
        return np.array(enlarged).reshape((len(list_) * multiple, len(list_[0]), -1))

    def resize_columns(list_: list) -> np.ndarray:
        enlarged = [[i] * multiple for column in list_ for i in column]
        return np.array(enlarged).reshape((len(list_), len(list_[0]) * multiple, -1))

    return resize_rows(resize_columns(maze_))


def maze_to_img(maze_: list, size_: int = CELL_SIZE) -> Image:
    maze_ = resize_maze(maze_, size_)
    pil_image = Image.fromarray(np.uint8(maze_)).convert('RGB')
    return pil_image


def maze_to_animation(maze_list: list, file_name: str, size_: int = CELL_SIZE, time_: int = 200) -> None:
    imgs = []
    for maze_ in maze_list:
        imgs.append(maze_to_img(maze_, size_))

    imgs[0].save(file_name + '.gif', save_all=True, append_images=imgs[1:], optimize=False, duration=time_)


def is_even(number: int) -> bool:
    if number % 2 == 0:
        return True
    return False


def correct_size(value: int) -> int:
    if is_even(value):
        return value - 1
    else:
        return value


def validate_size(w: int, h: int) -> tuple[int, int]:
    if (w >= 5) and (h >= 5):
        return correct_size(w), correct_size(h)
    else:
        raise ValueError("Maze size needs to be at least 5x5.")


def init_maze(width: int, height: int) -> tuple[int, int, list[list[list[int, int, int]]]]:
    width_, height_ = validate_size(width, height)
    return width_, height_, [[COLOUR_GRAY] * width_ for _ in range(height_)]


def build_maze(size_x: int, size_y: int) -> list[list[list[list[int, int, int]]]]:
    size_x, size_y, maze_ = init_maze(size_x, size_y)
    in_cells = []
    frontier_cells = []
    paths_to_next_cell = []
    build_steps = [copy.deepcopy(maze_)]

    def is_carved(xxx, yyy):
        if (xxx, yyy) in in_cells:
            return True
        return False

    def is_near_right_edge(pos):
        return pos == size_x - 2

    def is_near_left_edge(pos):
        return pos == 1

    def is_near_bottom_edge(pos):
        return pos == size_y - 2

    def is_near_top_edge(pos):
        return pos == 1

    def is_marked(xxx, yyy):
        if (xxx, yyy) in frontier_cells:
            return True
        return False

    def save_frontier(xxx, yyy, delta_x, delta_y):
        if not is_marked(xxx + 2*delta_x, yyy + 2*delta_y):
            frontier_cells.append((xxx + 2*delta_x, yyy + 2*delta_y))
            maze_[yyy + 2*delta_y][xxx + 2*delta_x] = COLOUR_YELLOW
        maze_[yyy + delta_y][xxx + delta_x] = COLOUR_YELLOW2

    def mark_cell(cx, cy):
        maze_[cy][cx] = COLOUR_WHITE2

    def save_cell(xxx, yyy):
        mark_cell(xxx, yyy)
        in_cells.append((xxx, yyy))

    def save_next_anim_frame():
        build_steps.append(copy.deepcopy(maze_))

    def mark_new_frontiers(fx, fy):
        if not is_near_right_edge(fx) and not is_carved(fx + 2, fy): save_frontier(fx, fy, 1, 0)
        if not is_near_left_edge(fx) and not is_carved(fx - 2, fy): save_frontier(fx, fy, -1, 0)
        if not is_near_bottom_edge(fy) and not is_carved(fx, fy + 2): save_frontier(fx, fy, 0, 1)
        if not is_near_top_edge(fy) and not is_carved(fx, fy - 2): save_frontier(fx, fy, 0, -1)

    def choose_a_random_element(list_: list[tuple[int, int]]) -> tuple[int, int]:
        shuffle(list_)
        return list_.pop()

    def choose_next_cell_from_frontiers() -> tuple[int, int]:
        return choose_a_random_element(frontier_cells)

    def get_neighbours_of_next_cell(xx, yy):
        if not is_near_right_edge(xx) and is_carved(xx + 2, yy): paths_to_next_cell.append((xx + 1, yy))
        if not is_near_left_edge(xx) and is_carved(xx - 2, yy): paths_to_next_cell.append((xx - 1, yy))
        if not is_near_bottom_edge(yy) and is_carved(xx, yy + 2): paths_to_next_cell.append((xx, yy + 1))
        if not is_near_top_edge(yy) and is_carved(xx, yy - 2): paths_to_next_cell.append((xx, yy - 1))

    def choose_a_neighbour() -> tuple[int, int]:
        return choose_a_random_element(paths_to_next_cell)

    def add_path_to_neighbour(px: int, py: int) -> None:
        mark_cell(px, py)

    def solidify_walls():
        while paths_to_next_cell:
            path_x, path_y = paths_to_next_cell.pop()
            maze_[path_y][path_x] = COLOUR_GRAY

    def add_head_and_tail(c: Cell) -> None:
        while True:
            x, y = ((size_x - 1) * c, randrange(1, size_y - 1))
            if is_carved(x + 1 - 2*c, y):
                mark_cell(x, y)
                save_next_anim_frame()
                break

    def add_maze_start_point():
        add_head_and_tail(Cell.START)

    def add_maze_end_point():
        add_head_and_tail(Cell.END)

    def carve_passage(xx: int, yy: int) -> None:
        while True:
            save_cell(xx, yy)
            save_next_anim_frame()

            mark_new_frontiers(xx, yy)
            save_next_anim_frame()

            if not frontier_cells:
                break
            else:
                xx, yy = choose_next_cell_from_frontiers()

            get_neighbours_of_next_cell(xx, yy)
            if paths_to_next_cell:
                path_x, path_y = choose_a_neighbour()
                add_path_to_neighbour(path_x, path_y)

                solidify_walls()

    x, y = (randrange(1, size_x - 1, 2), randrange(1, size_y - 1, 2))
    carve_passage(x, y)

    add_maze_start_point()
    add_maze_end_point()

    return build_steps


if __name__ == '__main__':
    size = input("Enter maze size as \"width\"x\"height\": ")
    w, h = size.split('x')
    maze = build_maze(int(w), int(h))
    maze_to_img(maze[-1], 10).show()
    maze_to_animation(maze, 'maze_big9', 10, 120)
