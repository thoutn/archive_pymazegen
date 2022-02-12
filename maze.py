import numpy as np
import numpy.typing as npt
from PIL import Image
from pprint import pprint
from random import randrange, shuffle
import copy


CELL_SIZE = 20


def resize_maze(maze_: list, multiple: int = CELL_SIZE) -> np.ndarray:
    def resize_rows(list_: np.ndarray) -> np.ndarray:
        enlarged = [[row] * multiple for row in list_]
        return np.array(enlarged).reshape((len(list_) * multiple, len(list_[0])))

    def resize_columns(list_: list) -> np.ndarray:
        enlarged = [[i] * multiple for column in list_ for i in column]
        return np.array(enlarged).reshape((len(list_), len(list_[0]) * multiple))

    return resize_rows(resize_columns(maze_))


def maze_to_img(maze_: np.ndarray) -> None:
    pil_image = Image.fromarray(np.uint8(maze_) * 255).convert('P')
    return pil_image


def maze_to_animation(maze_list: list, file_name: str, multiple: int = CELL_SIZE) -> None:
    imgs = []

    for maze_ in maze_list:
        imgs.append(maze_to_img(resize_maze(maze_, multiple)))

    imgs[0].save(file_name + '.gif', save_all=True, append_images=imgs[1:], optimize=False, duration=200)


def is_even(number: int) -> bool:
    if number % 2 == 0:
        return True
    return False


def correct_size(value: int) -> int:
    if is_even(value):
        return value - 1
    else:
        return value


def validate_size(w: int, h: int) -> tuple:
    if (w >= 5) and (h >= 5):
        return correct_size(w), correct_size(h)
    else:
        raise ValueError("Maze size needs to be at least 5x5.")


def init_maze(width: int, height: int) -> list:
    width_, height_ = validate_size(width, height)
    return width_, height_, [[0] * width_ for _ in range(height_)]


def build_maze(size_x: int, size_y: int) -> list:
    size_x, size_y, maze_ = init_maze(size_x, size_y)
    frontier_cells = []
    build_steps = [copy.deepcopy(maze_)]

    def is_carved(xxx, yyy):
        if maze_[yyy][xxx]:
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

    def get_next_element():
        return frontier_cells.pop()

    def carve_passage(xx: int, yy: int) -> None:
        while True:
            maze_[yy][xx] = 1
            build_steps.append(copy.deepcopy(maze_))

            if not is_near_right_edge(xx) and not is_carved(xx + 2, yy) and (xx + 2, yy) not in frontier_cells: frontier_cells.append((xx + 2, yy))
            if not is_near_left_edge(xx) and not is_carved(xx - 2, yy) and (xx - 2, yy) not in frontier_cells: frontier_cells.append((xx - 2, yy))
            if not is_near_bottom_edge(yy) and not is_carved(xx, yy + 2) and (xx, yy + 2) not in frontier_cells: frontier_cells.append((xx, yy + 2))
            if not is_near_top_edge(yy) and not is_carved(xx, yy - 2) and (xx, yy - 2) not in frontier_cells: frontier_cells.append((xx, yy - 2))

            if not frontier_cells:
                break
            else:
                shuffle(frontier_cells)
                xx, yy = get_next_element()

            wall_cells = []
            if not is_near_right_edge(xx) and is_carved(xx + 2, yy): wall_cells.append((xx + 1, yy))
            if not is_near_left_edge(xx) and is_carved(xx - 2, yy): wall_cells.append((xx - 1, yy))
            if not is_near_bottom_edge(yy) and is_carved(xx, yy + 2): wall_cells.append((xx, yy + 1))
            if not is_near_top_edge(yy) and is_carved(xx, yy - 2): wall_cells.append((xx, yy - 1))

            if wall_cells:
                shuffle(wall_cells)
                in_x, in_y = wall_cells.pop()
                maze_[in_y][in_x] = 1

    x, y = (randrange(1, size_x - 1, 2), randrange(1, size_y - 1, 2))
    carve_passage(x, y)

    while True:
        x, y = (0, randrange(1, size_y - 1))
        if is_carved(x + 1, y):
            maze_[y][x] = 1
            build_steps.append(copy.deepcopy(maze_))
            break

    while True:
        x, y = (size_x - 1, randrange(1, size_y - 1))
        if is_carved(x - 1, y):
            maze_[y][x] = 1
            build_steps.append(copy.deepcopy(maze_))
            break

    return build_steps


if __name__ == '__main__':
    size = input("Enter maze size as \"width\"x\"height\": ")
    w, h = size.split('x')
    maze = build_maze(int(w), int(h))
    #pprint(maze[-1])
    #pprint(maze[0])
    #pprint(resize_maze(maze[-1], 2))
    maze_to_img(resize_maze(maze[-1], 10)).show()
    maze_to_animation(maze, 'maze_big5', 20)
