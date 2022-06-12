# pymazegen (*archived*)

A maze generator written in Python 3.9. The project contains several experimental 
implementations of various algorithms to generate mazes.  

## Folder structure

There are various implementations of the maze generator. The project folder is structured as follows: 
- The root folder contains implementation of some algorithms 
([Recursive backtracker](https://en.wikipedia.org/wiki/Maze_generation_algorithm#Randomized_depth-first_search),
[Prim's](https://en.wikipedia.org/wiki/Prim%27s_algorithm),
[Kruskal's](https://en.wikipedia.org/wiki/Kruskal%27s_algorithm), 
[Hunt-and-kill](http://www.astrolog.org/labyrnth/algrithm.htm), 
[Eller's](http://www.neocomputer.org/projects/eller.html), 
[Backterial growth / Growing tree](https://weblog.jamisbuck.org/2011/1/27/maze-generation-growing-tree-algorithm.html)). 
The code enables to visualise the building steps of each algo, i.e. generates animation. 
- [`mods/`](mods/) is the home of various static (no animation) implementations of the maze generator. 
It contains more algorithms than the root folder (unfortunately, in an uncomfortably packed single[^1] file). 
- [`mods/cython_code/`](mods/cython_code/), as its name states, contains a **Cython** 'optimized' (experiment) version of `maze_nng.py`. 

### More on `mods/`

Three representations of the maze have been tested:
1. `maze_m.py` - maze represented by a matrix. Both the passage and wall are represented in the matrix. Carved passages 
  are marked with a value of `1`, otherwise the matrix elements have a default value of `0` (walls). 
2. `maze_nng.py` - maze represented as a *graph*, with each cell being represented as a *node* 
  (class `Cell`) of the graph. More on this implementation in [this book][1]. 
3. `maze_mnd.py` - a hybrid representation of the maze. It uses a matrix, but the entries are of type `dict` rather 
  than `int` with values of `0` or `1` as in (1). The dictionary provides information whether the given cell is or is not linked 
  to its neighbours. 

[^1]: Actually there are three files with various representations of the maze. Each file contains several algorithms 
and the code to render a 2D image of the final maze.

[1]: http://www.mazesforprogrammers.com/ "Mazes for programmers"

## How to use the project

> **Note** *This project is archived - as written above, it includes all the experiments with the algorithms 
that were considered during the development of the project. A clean and final[^2] version of the project can be 
found [here](). You are strongly recommended to use that one.* 
> 
> *In case you would like to know what's the fuss about this one, please follow the information below.*

There are several files in the project folder - as described in the previous sections -, which can be used as 
the access point to a specific implementation of the maze generator. 

[^2]: Animation not yet implemented \[update 10-Jun-2022\]. 

### File `./main.py`

The root folder contains file `main.py`, which is a central file to access each code in the root folder.  

Function `main` is called on line ***51*** of `main.py`, with the following arguments:
- `w`: *int* - width of the maze, 
- `h`: *int* - height of the maze, 
- `algo`: *Algo* - the algorithm to generate a maze of size `w`×`h`. 
The following values are allowed for this parameter:
```
Algo.BACKTRACKER
Algo.PRIM
Algo.KRUSKAL
Algo.ELLER
Algo.HUNT_KILL
Algo.BACTERIAL
```
Changing the arguments on line ***51*** will generate a corresponding maze and animation of the build steps. 

### File `./mods/maze_nng.py`

This is the file where most of the experimentation have happened. 

The file contains various forms of the implemented algorithms 
([Binary Tree](http://weblog.jamisbuck.org/2011/2/1/maze-generation-binary-tree-algorithm.html),
[Recursive backtracker](https://en.wikipedia.org/wiki/Maze_generation_algorithm#Randomized_depth-first_search),
[Sidewinder](http://weblog.jamisbuck.org/2011/2/3/maze-generation-sidewinder-algorithm.html),
[Prim's](https://en.wikipedia.org/wiki/Prim%27s_algorithm),
[Kruskal's](https://en.wikipedia.org/wiki/Kruskal%27s_algorithm),
[Eller's](http://weblog.jamisbuck.org/2010/12/29/maze-generation-eller-s-algorithm.html),
[Hunt-and-kill](http://www.astrolog.org/labyrnth/algrithm.htm),
[Hunt-and-kill scan mode](http://weblog.jamisbuck.org/2011/1/24/maze-generation-hunt-and-kill-algorithm.html),
[AldousBroder](https://en.wikipedia.org/wiki/Maze_generation_algorithm#Aldous-Broder_algorithm),
[Wilson's](https://en.wikipedia.org/wiki/Maze_generation_algorithm#Wilson's_algorithm),
[Recursive division](https://en.wikipedia.org/wiki/Maze_generation_algorithm#Recursive_division_method)) 
The optimization results can be visualised by calling function `plot_statistics` as on line ***1207***. 
More about the optimization results [here](mods/README.md). 

A maze can be generated using either algorithm following the steps in the example below. 
```
grid = Grid(10, 10)
huntkill = HuntAndKillScan3MazeBuilder(grid)
huntkill.build_maze()
```
The generated maze can be rendered as an image using the example below. 
```
img = ImagePresenter(grid, wall_thickness=2)
img.render()
```
The file implements an Ascii renderer too. It can print the generated maze into the terminal. In case this option is preferred over 
the image renderer, please follow the steps in the example below. 
```
ascii_art = AsciiPresenter(grid)
ascii_art.to_string()
```
This and several other examples are included in the file, and can be found from line ***1055*** to ***1202***. 

### File `./mods/maze_mnd.py` and `./mods/maze_m.py`

Usage same as for `./mods/maze_nng.py`. 

### File `./mods/cymaze.py`

Usage same as above.

## Licence

[MIT License](LICENSE)