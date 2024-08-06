import numpy as np  # 只需要下载numpy库即可
import random
import GridWorld_v2

gamma = 0.999
rows = 5
columns = 5
forbiddenAreaScore = -10
gridworld = GridWorld_v2.GridWorld_v2(forbiddenAreaScore=forbiddenAreaScore, score=1,
                                      desc=[".....", ".##..", "..#..", ".#T#.", ".#..."])

np.random.seed(23)
value = np.zeros((rows * columns))
qtable = np.zeros((rows * columns, 5))
policy = np.eye(5)[np.random.randint(0, 5, size=(rows * columns))]
gridworld.show()
gridworld.showPolicy(policy)
print("random policy")

gridworld.show()  # 打印gridworld
gridworld.showPolicy(policy)

trajectorySteps = 20000
epsilon = 0.1
num_episodes = 6000
learning_rate = 0.01

import time

start_time = time.time()
for episode in range(num_episodes):
    epsilon = epsilon - 0.0001 if epsilon > 0.0001 else 0.0001
    p1 = 1 - epsilon * (4 / 5)
    p0 = epsilon / 5
    d = {1: p1, 0: p0}
    policy_epsilon = np.vectorize(d.get)(policy)

    trajectory = gridworld.getTrajectoryScore(nowState=np.random.randint(0, 24),
                                              action=np.random.randint(0, 4),
                                              policy=policy_epsilon, steps=trajectorySteps,
                                              stop_when_reach_target=True)
    trajectory.append((17, 4, 1, 17, 4))
    # if len(trajectory) > 5000:
    #     continue
    # value[17] = 100

    # policy evaluation
    for k in range(len(trajectory) - 1, -1, -1):
        now_state, now_action, now_score, next_state = trajectory[k][0:4]
        TD_target = now_score + gamma * value[next_state]
        TD_error = value[now_state] - TD_target
        value[now_state] = value[now_state] - learning_rate * TD_error
    for state in range(rows * columns):
        for action in range(5):
            score, next_state = gridworld.getScore(state, action)
            qtable[state][action] = score + gamma * value[next_state]

    # policy improvement
    policy = np.eye(5)[np.argmax(qtable, axis=1)]
    gridworld.showPolicy(policy)

print("All state value: ", np.sum(value))
end_time = time.time()
print("TD time:", end_time - start_time)
