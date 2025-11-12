"""
@file   : stock_max_profit.py
@time   : 2025-11-12
"""
# l1 = [[0, 0]] * 10
# print(l1)
# l1[0][0] = 1
# print(l1)
#
# l1 = [10, 20, [30, 40]]
# res = l1 * 5
# # print(res)
# res[0] = 100
# print(res)   # [100, 20, [30, 40], 10, 20, [30, 40], 10, 20, [30, 40], 10, 20, [30, 40], 10, 20, [30, 40]]
# res[2][0] = 300
# print(res)   # [100, 20, [300, 40], 10, 20, [300, 40], 10, 20, [300, 40], 10, 20, [300, 40], 10, 20, [300, 40]]

# 无限次交易。每次交易完后 必须停一天 才可以再交易
def maxProfit(prices):
    # [[0, 0]]
    # dp = [[0] * 2] * len(prices)
    dp = [[0, 0] for i in range(len(prices))]
    # print(dp)   # [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
    for i in range(len(prices)):
        if i == 0:
            dp[i][0] = 0
            dp[i][1] = -prices[i]
        elif i == 1:
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
            dp[i][1] = max(dp[i-1][1], dp[i-1][0] - prices[i])
        else:
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
            dp[i][1] = max(dp[i-1][1], dp[i-2][0] - prices[i])
    return dp[len(prices)-1][0]


prices = [1, 2, 3, 0, 2]
res = maxProfit(prices)
print(res)

# 第0天   无: 0                 有: -1
# 第1天   无: max(0, 1)=1       有: max(-1, -2) = -1
# 第2天   无: max(1, 2)=2       有: max(-1, -3) = -1
# 第3天   无: max(-1+0, 2)=2    有: max(-1, 1+) = 2
# ....

