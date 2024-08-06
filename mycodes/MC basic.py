import numpy as np
import random
import GridWorld_v2

gamma = 0.9  # 折扣因子，越接近0越近视
rows = 10  # 记得行数和列数这里要同步改
columns = 10
seed = 12
forbiddenAreaScore = -10
forbiddenAreaNums = 20

trajectorySteps = 5
gridworldv2 = GridWorld_v2.GridWorld_v2(rows=rows, columns=columns,
                                        forbiddenAreaScore=forbiddenAreaScore,
                                        forbiddenAreaNums=forbiddenAreaNums,
                                        score=1,
                                        # desc=[".....", ".##..", "..#..", ".#T#.", ".#..."],
                                        seed=seed,
                                        )
value = np.zeros(rows * columns)  # 初始化可以任意，也可以全0
qtable = np.zeros((rows * columns, 5))  # 初始化，这里主要是初始化维数，里面的内容会被覆盖所以无所谓
policy = np.eye(5)[np.random.randint(0, 5, size=(rows * columns))]  # 初始策略
# policy = np.random.randint(0, 5, size=(rows, columns))
gridworldv2.show()  # 打印gridworld
gridworldv2.showPolicy(policy)

np.set_printoptions(precision=2, suppress=True)
qtable_pre = qtable.copy() + 1  # 随便弄个不一样的初值

import time

start_time = time.time()
mc_basic_policy = policy
while np.sum((qtable_pre - qtable) ** 2) > 0.001:
    # print(np.sum((qtable_pre - qtable) ** 2))
    qtable_pre = qtable.copy()

    # policy evaluation
    for state in range(rows * columns):
        for action in range(5):
            Trajectory = gridworldv2.getTrajectoryScore(nowState=state, action=action, policy=mc_basic_policy,
                                                        steps=trajectorySteps)
            acc_reward = Trajectory[trajectorySteps][2]
            for k in range(trajectorySteps - 1, -1, -1):
                acc_reward = acc_reward * gamma + Trajectory[k][2]
            qtable[state][action] = acc_reward
        # policy improvement
        # 此处因为模拟trajectory依赖于policy, 因此每轮都要更新policy
        mc_basic_policy = np.eye(5)[np.argmax(qtable, axis=1)]  # 此处为一个state更新一次，也可以一次完整的exploring starts更新一次
    # show value and policy
    gridworldv2.showPolicy(mc_basic_policy)
    value = np.max(qtable * mc_basic_policy, axis=1)
    print(value.reshape(rows, columns))

print("All state value: ", np.sum(value))
end_time = time.time()
print("MC basic time:", end_time - start_time)
