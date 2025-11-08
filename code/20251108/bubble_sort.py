"""
@file   : bubble_sort.py
@time   : 2025-11-08
"""

def run_bubble(l1):
    for i in range(len(l1)-1, 0, -1):
        # i 控制的比较次数
        for j in range(i):
            if l1[j] > l1[j+1]:
                l1[j], l1[j+1] = l1[j+1], l1[j]
    return l1


l1 = [42, 12, 32, 65, 75, 12, 69, 100, 43]  # 从小到大排序  # 不能用sort
res = run_bubble(l1)
print(res)  # [12, 12, 32, 42, 43, 65, 69, 75, 100]

