"""
@file   : 006.py
@time   : 2025-11-19
"""

def func1(l1):
    result = []
    n = len(l1)

    l1.sort()
    # [False False False]
    def backtrack(nums, tmp, is_visited):
        if len(tmp) == n:
            result.append(tmp.copy())
            return

        for i in range(len(nums)):
            if is_visited[i]:
                continue

            if i > 0 and nums[i] == nums[i-1] and not is_visited[i-1]:
                continue

            is_visited[i] = True
            tmp.append(nums[i])
            backtrack(nums, tmp, is_visited)
            is_visited[i] = False
            tmp.pop()

    is_visited = [False] * len(l1)
    backtrack(l1, [], is_visited)
    return result

l1 = [1, 1, 3]  # 全排列  [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2,3,1],[3,1,2], [3,2,1]]
res = func1(l1)
print(res)



