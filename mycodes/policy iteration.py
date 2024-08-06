import numpy as np
import random
import GridWorld_v1

gamma = 0.9  # 折扣因子，越接近0越近视
rows = 10  # 记得行数和列数这里要同步改
columns = 10
seed = 12
forbiddenAreaScore = -10
forbiddenAreaNums = 20

gridworldv1 = GridWorld_v1.GridWorld_v1(rows=rows, columns=columns,
                                        forbiddenAreaScore=forbiddenAreaScore,
                                        forbiddenAreaNums=forbiddenAreaNums,
                                        score=1,
                                        # desc=[".....", ".##..", "..#..", ".#T#.", ".#..."],
                                        # seed=random.randint(1, 1000),
                                        seed=seed,
                                        )
value = np.zeros(rows * columns)  # 初始化可以任意，也可以全0
qtable = np.zeros((rows * columns, 5))  # 初始化，这里主要是初始化维数，里面的内容会被覆盖所以无所谓
policy = np.argmax(qtable, axis=1)  # 初始策略
gridworldv1.show()  # 打印gridworld
gridworldv1.showPolicy(policy)

np.set_printoptions(precision=2, suppress=True)
pre_value = value.copy() + 1  # 随便弄个不一样的初值

import time

start_time = time.time()
pol_iter_policy = policy
while np.sum((pre_value - value) ** 2) > 0.001:
    # print("Euclidean Distance:", np.sum((pre_value - value) ** 2))  #计算欧几里得距离
    pre_value = value.copy()  # 保存副本，方便前后对比

    # policy evaluation
    pre_value_0 = value.copy() + 1
    while np.sum((pre_value_0 - value) ** 2) > 0.001:
        # 此while循环实际是在解贝尔曼方程求value
        # 因为就算model-based的情况下，也很难直接求value的解析解
        # 所以需要用迭代的方法求解value
        pre_value_0 = value.copy()
        for state in range(rows * columns):
            action = np.argmax(qtable[state])
            score, nextState = gridworldv1.getScore(state, action)
            # 由qtable/policy更新value
            value[state] = score + gamma * value[nextState]
    # 求value本质是为了求qtable，因为argmax(qtable) -> policy
    # 而MC basic中可以直接模拟得到qtable，因此可以无需求value
    for state in range(rows * columns):
        for action in range(5):
            score, nextState = gridworldv1.getScore(state, action)
            # 本质上更新q就是在更新policy
            qtable[state][action] = score + gamma * value[nextState]

    # policy improvement
    pol_iter_policy = np.argmax(qtable, axis=1)
    # show value and policy
    gridworldv1.showPolicy(pol_iter_policy)
    print(value.reshape(rows, columns))

print("All state value: ", np.sum(value))
end_time = time.time()
print("Policy Iteration time:", end_time - start_time)
