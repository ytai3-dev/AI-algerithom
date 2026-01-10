"""
@file   : 002.py
@time   : 2025-11-19
"""

# [[1, 2, 3], [4, 5], [7, 8]]
# 求出: [147, 148, 157, 158, 247, 248, 347, 348]

def func1(l1):
    result = []
    n = len(l1)

    def backtrack(nums, tmp):
        if len(tmp) == n:
            result.append(tmp.copy())
            return

        for v in nums[0]:
            backtrack(nums[1:], tmp + [v])

    backtrack(l1, [])
    return result


l1 = [[1, 2, 3], [4, 5], [7, 8]]
res = func1(l1)
print(res)

