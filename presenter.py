from abc import ABC, abstractmethod

import numpy as np
from PIL import Image

CELL_SIZE = 20
WALL_THICKNESS = 2


class MazePresenter(ABC):
    def __init__(self, maze_: list, multiple: int = CELL_SIZE):
        self.maze_ = maze_
        self.multiple = multiple

    @abstractmethod
    def resize_maze(self, matrix_: list):
        pass

    def maze_to_img(self, matrix_: list = None) -> Image:
        if not matrix_:
            matrix_ = self.maze_[-1]
        matrix_ = self.resize_maze(matrix_)
        pil_image = Image.fromarray(np.uint8(matrix_)).convert('RGB')
        return pil_image

    def maze_to_animation(self, file_name: str, time_: int = 200) -> None:
        imgs = []
        for m_ in self.maze_:
            imgs.append(self.maze_to_img(m_))

        imgs[0].save(file_name + '.gif', save_all=True, append_images=imgs[1:], optimize=False, duration=time_)


class MazePresenterThickStyle(MazePresenter):
    def resize_maze(self, maze_: list) -> np.ndarray:
        def resize_rows(container: np.ndarray) -> np.ndarray:
            return np.repeat(container, repeats=self.multiple, axis=0)

        def resize_columns(container: list) -> np.ndarray:
            return np.repeat(container, repeats=self.multiple, axis=1)

        return resize_rows(resize_columns(maze_))


class MazePresenterThinStyle(MazePresenter):
    def resize_maze(self, maze_: list) -> np.ndarray:
        def resize_rows(container: np.ndarray) -> np.ndarray:
            img = np.repeat(container, repeats=self.multiple, axis=0)
            for i in range(len(container)):
                if i % 2 == 0:
                    rows = [-(x + i * self.multiple - i // 2 * (self.multiple - WALL_THICKNESS)) for x in
                            range(1, self.multiple - WALL_THICKNESS + 1)]
                    img = np.delete(img, tuple(rows), axis=0)
            return img

        def resize_columns(container: list) -> np.ndarray:
            img = np.repeat(container, repeats=self.multiple, axis=1)
            for i in range(len(container[0])):
                if i % 2 == 0:
                    cols = [-(x + i * self.multiple - i // 2 * (self.multiple - WALL_THICKNESS)) for x in
                            range(1, self.multiple - WALL_THICKNESS + 1)]
                    img = np.delete(img, tuple(cols), axis=1)
            return img

        return resize_rows(resize_columns(maze_))