from enum import IntEnum
from typing import Callable

from maze_backtracking import *
from maze_prims import *
from maze_kruskals import *
from maze_ellers import *
from maze_huntkill import *
from maze_bacterial_growth import *

from presenter import MazePresenterThinStyle, MazePresenterThickStyle


class Algo(IntEnum):
    BACKTRACKER = 0
    PRIM = 1
    KRUSKAL = 2
    ELLER = 3
    HUNT_KILL = 4
    BACTERIAL = 5


def choose_builder(algo: Algo) -> Callable:
    return (
        RecursiveBacktrackingMazeBuilder,
        RandomisedPrimsMazeBuilder,
        RandomisedKruskalsMazeBuilder,
        EllersMazeBuilder,
        HuntAndKillMazeBuilder,
        BacterialGrowthMazeBuilder,
    )[algo]


def main(w: int, h: int, algo: Algo) -> None:
    builder = choose_builder(algo)

    maze = builder(w, h)
    maze.build_maze()

    thick_walls = MazePresenterThickStyle(maze.build_steps, 10)
    thick_walls.maze_to_img().show()

    thin_walls = MazePresenterThinStyle(maze.build_steps)
    thin_walls.maze_to_img().show()

    thin_walls.maze_to_animation('maze_', 120)


if __name__ == '__main__':
    # Prim's algo
    main(30, 30, Algo.PRIM)
