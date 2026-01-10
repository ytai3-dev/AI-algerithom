"""
@file   : 004.py
@time   : 2025-11-19
"""
def letterCombinations(digits):
    d = {2: ['a', 'b', 'c'], 3: ['d', 'e', 'f'], 4: ['g', 'h', 'i'], 5: ['j', 'k', 'l'], 6: ['m', 'n', 'o'],
         7: ['p', 'q', 'r', 's'], 8: ['t', 'u', 'v'], 9: ['w', 'x', 'y', 'z']}

    result = []
    for v in digits:
        temp = d[int(v)]
        result.append(temp)
    # print(result)  # [['a', 'b', 'c'], ['d', 'e', 'f']]
    final_result = []
    n = len(result)
    def backtrack(nums, temp):
        if len(temp) == n:
            final_result.append(''.join(temp.copy()))
            return

        for v in nums[0]:
            backtrack(nums[1:], temp + [v])

    backtrack(result, [])
    return final_result

res = letterCombinations('23')
print(res)

exit()









"""
# {"2": ['a', 'b', 'c'], '3': ['d', 'e', 'f'], ...}
result = []
for i in range(26):
    result.append(chr(97 + i))
# print(result)  # ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

d = {}
for i in range(2, 10):
    # i=2   0:3     (i-2)*3:(i-1)*3
    # i=3   3:6
    # i=4   6:9
    # ...

    if i < 7:
        d[i] = result[(i-2)*3: (i-1)*3]
    elif i==7:
        d[i] = result[(i-2)*3: (i-2)*3+4]
    elif i==8:
        # i=8
        d[i] = result[19:22]
    else:
        d[i] = result[22:]
print(d)
"""