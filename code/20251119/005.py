"""
@file   : 005.py
@time   : 2025-11-19
"""
result = [[1,1,2], [1,2,1], [2,1,1], [1, 2, 1], [1, 1, 2]]
# res = set(result)
# print(res)

res = []
for v in result:
    if v not in res:
        res.append(v)
print(res)
