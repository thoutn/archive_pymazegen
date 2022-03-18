from maze_huntkill import HuntAndKillMazeBuilder
from presenter import MazePresenterThickStyle, MazePresenterThinStyle

if __name__ == '__main__':
    size = input("Enter maze size as \"width\"x\"height\": ")
    w, h = size.split('x')

    prims_maze = RandomisedPrimsMazeBuilder(int(w), int(h))
    prims_maze.build_maze()

    thick_walls = MazePresenterThickStyle(prims_maze.build_steps, 10)
    thin_walls = MazePresenterThinStyle(prims_maze.build_steps, 10)
    thick_walls.maze_to_img().show()
    thin_walls.maze_to_img().show()
    thin_walls.maze_to_animation('maze_prims', 120)

    kruskals_maze = RandomisedKruskalsMazeBuilder(int(w), int(h))
    kruskals_maze.build_maze()

    thin_walls = MazePresenterThinStyle(kruskals_maze.build_steps, 10)
    thin_walls.maze_to_img().show()
    thin_walls.maze_to_animation('maze_kruskals', 120)

    backtracker_maze = RecursiveBacktrackingMazeBuilder(int(w), int(h))
    backtracker_maze.build_maze()

    thin_walls = MazePresenterThinStyle(backtracker_maze.build_steps, 10)
    thin_walls.maze_to_img().show()
    thin_walls.maze_to_animation('maze_backtracker', 120)

    huntkill_maze = HuntAndKillMazeBuilder(int(w), int(h))
    huntkill_maze.build_maze()

    thin_walls = MazePresenterThinStyle(huntkill_maze.build_steps, 10)
    thin_walls.maze_to_img().show()
    thin_walls.maze_to_animation('maze_huntkill', 120)

    bacteria_maze = BacterialGrowthMazeBuilder(int(w), int(h))
    bacteria_maze.build_maze()

    thin_walls = MazePresenterThinStyle(bacteria_maze.build_steps, 10)
    thin_walls.maze_to_img().show()
    thin_walls.maze_to_animation('maze_bacteria', 120)

    ellers_maze = EllersMazeBuilder(int(w), int(h))
    ellers_maze.build_maze()

    thin_walls = MazePresenterThinStyle(ellers_maze.build_steps, 20)
    thin_walls.maze_to_img().show()
    thin_walls.maze_to_animation('maze_ellers', 120)
