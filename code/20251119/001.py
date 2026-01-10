"""
@file   : run_train.py
@time   : 2025-11-19
"""
import numpy


#
# def func1(l1):
#     result = []
#     for v in l1:
#         # 判断v是不是列表
#         if type(v) == list:
#             result.extend(func1(v))
#         else:
#             result.append(v)
#     return result
#
# # 将列表中所有值取出

def func1(l1):
    result = []

    def backtrack(nums):
        if type(nums) != list:
            result.append(nums)
            return

        for v in nums:
            backtrack(v)
    backtrack(l1)
    return result

l1 = [[1], [2, [3], [4, 5]], [7, 8, [9], [10, 11]], [12], 13]
res = func1(l1)
print(res)


def func1(l1):
    result = []
    n = len(l1)

    def backtrack(nums, tmp):
        if len(tmp) == n:
            result.append(tmp.copy())
            return

        for i in range(len(nums)):
            backtrack(nums[:i] + nums[i+1:], tmp+[nums[i]])
    backtrack(l1, [])
    print(result)

l1 = [1, 2, 3]  # 全排列  [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2,3,1],[3,1,2], [3,2,1]]
func1(l1)


'''
backtrack([1, 2, 3], [])
    i=0    backtrack([2, 3], [1]) 
                 i=0       backtrack([3], [1, 2])
                                i=0       backtrack([], [1, 2, 3])      result=[[1, 2, 3]]
                 i=1       backtrack([2], [1, 3])
                                i=0       backtrack([], [1, 3, 2])      result=[[1,2,3], [1,3,2]]
    i=1    backtrack([1, 3], [2])
                 i=0       backtrack([3], [2,1])
                                i=0       backtrack([], [2, 1, 3])      result=[[1,2,3], [1,3,2], [2,1,3]
                 i=1       backtrack([1], [2,3]) 
                                i=0       backtrack([], [2, 3, 1])      result=[[1,2,3], [1,3,2], [2,1,3], [2, 3,1]]  
                                
'''
