"""
@file   : 003.py
@time   : 2025-11-19
"""


def func1(grid):
    row_num = len(grid)
    col_num = len(grid[0])

    def backtrack(r, c):
        if r < 0 or r >= row_num or c < 0 or c >= col_num or grid[r][c] == 0:
            return 0

        grid[r][c] = 0
        area = 1
        for x, y in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            dr = r+x
            dc = c+y
            area += backtrack(dr, dc)
        return area

    max_area = -float('inf')
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == 1:
                # 算面积
                temp = backtrack(r, c)
                max_area = max(max_area, temp)
    return max_area


grid = [[0,0,1,0,0,0,0,1,0,0,0,0,0],
        [0,0,0,0,0,0,0,1,1,1,0,0,0],
        [0,1,1,0,1,0,0,0,0,0,0,0,0],
        [0,1,0,0,1,1,0,0,1,0,1,0,0],
        [0,1,0,0,1,1,0,0,1,1,1,0,0],
        [0,0,0,0,0,0,0,0,0,0,1,0,0],
        [0,0,0,0,0,0,0,1,1,1,0,0,0],
        [0,0,0,0,0,0,0,1,1,0,0,0,0]]

res = func1(grid)
print(res)


