"""
@file   : select_sort.py
@time   : 2025-11-08
"""
def run_select_sort(l1):
    for i in range(len(l1)):
        min_value_index = i
        for j in range(i+1, len(l1)):
            if l1[j] < l1[min_value_index]:
                min_value_index = j
        if min_value_index != i:
            l1[i], l1[min_value_index] = l1[min_value_index], l1[i]
    return l1


l1 = [42, 12, 32, 65, 75, 12, 69, 100, 43]  # 从小到大排序  # 不能用sort
res = run_select_sort(l1)
print(res)  # [12, 12, 32, 42, 43, 65, 69, 75, 100]

