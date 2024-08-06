import numpy as np
import random
import GridWorld_v2

gamma = 0.9  # 折扣因子，越接近0越近视
rows = 5  # 记得行数和列数这里要同步改
columns = 5
seed = 12
forbiddenAreaScore = -10
forbiddenAreaNums = 20

gridworldv2 = GridWorld_v2.GridWorld_v2(rows=rows, columns=columns,
                                        forbiddenAreaScore=forbiddenAreaScore,
                                        forbiddenAreaNums=forbiddenAreaNums,
                                        score=1,
                                        desc=[".....", ".##..", "..#..", ".#T#.", ".#..."],
                                        # seed=seed,
                                        )
value = np.zeros(rows * columns)  # 初始化可以任意，也可以全0
qtable = np.zeros((rows * columns, 5))  # 初始化，这里主要是初始化维数，里面的内容会被覆盖所以无所谓
policy = np.eye(5)[np.random.randint(0, 5, size=(rows * columns))]  # 初始策略
# policy = np.random.randint(0, 5, size=(rows, columns))
gridworldv2.show()  # 打印gridworld
gridworldv2.showPolicy(policy)

np.set_printoptions(precision=2, suppress=True)
qtable_pre = qtable.copy() + 1  # 随便弄个不一样的初值

trajectorySteps = 20000
epsilon = 0.1
num_episodes = 200

import time

start_time = time.time()
for epsisode in range(num_episodes):
    epsilon = epsilon - 0.001 if epsilon > 0.001 else 0.001
    p1 = 1 - epsilon * (4/5)
    p0 = epsilon / 5
    d = {1: p1, 0: p0}
    policy_epsilon = np.vectorize(d.get)(policy)

    # 记录every visit中可能的重复(state,action)的信息
    # qtable_rewards = [[0 for _ in range(5)]] * (rows * columns)
    # qtable_nums = [[0 for _ in range(5)]] * (rows * columns)

    # data collection
    trajectory = gridworldv2.getTrajectoryScore(nowState=random.randint(0, 24), action=random.randint(0, 4),
                                                policy=policy_epsilon, steps=trajectorySteps)

    # policy evaluation
    acc_reward = 0
    for k in range(trajectorySteps, -1, -1):
        now_state, now_action, now_score = trajectory[k][0:3]
        acc_reward = acc_reward * gamma + now_score
        # every visit
        # qtable_rewards[now_state][now_action] += acc_reward
        # qtable_nums[now_state][now_action] += 1
        # qtable[now_state][now_action] = qtable_rewards[now_state][now_action] / qtable_nums[now_state][now_action]

        # first visit
        # 相同的(state,value)出现在一个trajectory中的时候，只用第一个信息
        qtable[now_state][now_action] = acc_reward

    # values = []
    # for state in range(rows * columns):
    #     v = 0
    #     for action in range(5):
    #         v += policy_epsilon[state][action] * qtable[state][action]
    #     values.append(v)
    # print(np.array(values).reshape(5, -1))

    # policy improvement
    policy = np.eye(5)[np.argmax(qtable, axis=1)]
    gridworldv2.showPolicy(policy)

print("All state value: ", np.sum(value))
end_time = time.time()
print("MC e-greed time:", end_time - start_time)
