import matplotlib.pyplot as plt
import timeit
import statistics
from PIL import Image, ImageDraw


from cython_code.maze_nng_cy import Grid


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
        w = self._set_size(self.grid.width)
        h = self._set_size(self.grid.height)

        img = Image.new("RGB", size=(w, h), color=(220, 220, 220))
        draw = ImageDraw.Draw(img)

        for row in self.grid.cells:
            for cell in row:
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
                    shape_ = [(w // 2 - radius1 + wall_width, h // 2 - radius1 + wall_width), (w // 2 + radius1 - wall_width, h // 2 + radius1 - wall_width)]
                    draw.arc(shape_, start=(col * theta) + alpha, end=((col + 1) * theta) - alpha, fill=(250, 150, 150), width=self.size * ANTIALIAS_ - wall_width)

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
        # global base_time
        x = []
        y = []

        for i in range(20, 100, 10):
            if "maze_nng_cy" in method:
                stmt_code = f'grid = Grid({i}, {i}) \nmethod = {method}MazeBuilder(grid) \nmethod.build_maze()'
                time = timeit.repeat(stmt=stmt_code,
                                     setup=f'from cython_code.maze_nng_cy import Grid \n'
                                           + f'import cython_code.maze_nng_cy as maze_nng_cy',
                                     repeat=5,
                                     number=10)
            elif "maze_nng_py" in method:
                stmt_code = f'grid = Grid({i}, {i}) \nmethod = {method}MazeBuilder(grid) \nmethod.build_maze()'
                time = timeit.repeat(stmt=stmt_code,
                                     setup=f'from cython_code.maze_nng_py import Grid \n'
                                           + f'import cython_code.maze_nng_py as maze_nng_py',
                                     repeat=5,
                                     number=10)
            else:
                stmt_code = f'grid = Grid({i}, {i}) \nmethod = {method}MazeBuilder(grid) \nmethod.build_maze()'
                time = timeit.repeat(stmt=stmt_code,
                                     setup=f'from maze_nng import Grid \nimport maze_nng',
                                     repeat=5,
                                     number=10)
            x.append(i)
            y.append(statistics.mean(time))# / base_time)

        plt.plot(x, y, label=method, c=next(colors), ls=next(markers))

    plt.legend(loc="upper left")
    plt.show()


if __name__ == '__main__':
    import maze_nng
    # plot_statistics(["maze_nng_cy.BinaryTree",
    #                  "maze_nng_py.BinaryTree",
    #                  "maze_nng.BinaryTree",
    #                  # "RecursiveBacktracker",
    #                  # "Sidewinder",
    #                  # # # "Prims_old",
    #                  # # "Prims",
    #                  # # # "Kruskals",
    #                  # # "Ellers",
    #                  # # "HuntAndKill",
    #                  # # # "HuntAndKillScan",
    #                  # # "AldousBroder",
    #                  # # "Wilsons",
    #                  # # # "HuntAndKillScan2",
    #                  #  "Kruskals2",
    #                  # # "HuntAndKillScan3",
    #                  "maze_nng_cy.Kruskals3",
    #                  "maze_nng_py.Kruskals3",
    #                  "maze_nng.Kruskals3"])
    #                  # "RecursiveDivision"])

    from cython_code.maze_nng_cy import BinaryTreeMazeBuilder

    # Binary Tree algo
    grid = Grid(10, 10)
    binary = BinaryTreeMazeBuilder(grid)
    binary.build_maze()

    img = ImagePresenter(grid, wall_thickness=2)
    img.render()
