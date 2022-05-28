# pymazegen (*archived*)

---

A maze generator written in Python. The project contains several experimental 
implementations of various algorithms to generate mazes.  

## Folder structure

---

There are various implementations of the maze generator. The project folder is structured as follows: 
- The root folder contains implementation of some algorithms 
([Recursive backtracker](https://en.wikipedia.org/wiki/Maze_generation_algorithm#Randomized_depth-first_search),
[Prim's](https://en.wikipedia.org/wiki/Prim%27s_algorithm),
[Kruskal's](https://en.wikipedia.org/wiki/Kruskal%27s_algorithm), 
[Hunt and kill](http://www.astrolog.org/labyrnth/algrithm.htm), 
[Eller's](http://www.neocomputer.org/projects/eller.html), 
[Backterial growth / Growing tree](https://weblog.jamisbuck.org/2011/1/27/maze-generation-growing-tree-algorithm.html)). 
The code enables to visualise the building steps of each algo. 
- The `/mods` folder is the home of various static (no animation) implementations of the maze generator. 
It contains more algorithms than the root folder (unfortunately, in an uncomfortably packed single[^1] file). 
- `/mods/cython_code`, as its name states, contains a **Cython** optimized (experiment) version of `maze_nng.py`. 

### More on `/mods`

Three representations of the maze have been tested:
  - `maze_m.py` - maze represented by a matrix. Both the passage and wall are present in the matrix. Carved passages 
  are marked with a value of `1`, otherwise the matrix elements have a default value of `0` (walls). 
  - `maze_nng.py` - maze represented as a *graph*, with each cell being represented as a *node* 
  (class `Cell`) of the graph. 
  - `maze_mnd.py` - a hybrid representation of the maze. It uses a matrix, but the entries are of type `dict` rather 
  than `int` with values of `0` or `1`. The dictionary provides information whether the given cell is or is not linked 
  to its neighbours. 

[^1]: Actually there are three files with various representations of the maze. Each file contains several algorithms 
and the code to render a 2D image of the final maze.

## How to use the project

---

