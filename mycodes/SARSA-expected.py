import numpy as np
import random
import GridWorld_v2

gamma = 0.99
rows = 5
columns = 5
forbiddenAreaScore = -10
gridworld = GridWorld_v2.GridWorld_v2(forbiddenAreaScore=forbiddenAreaScore, score=1,
                                      desc=[".....", ".##..", "..#..", ".#T#.", ".#..."])
value = np.zeros((rows * columns))
qtable = np.zeros((rows * columns, 5))
policy = np.eye(5)[np.random.randint(0, 5, size=(rows * columns))]

gridworld.show()  # 打印gridworld
gridworld.showPolicy(policy)

trajectorySteps = 20000
epsilon = 0.5
num_episodes = 600
learning_rate = 0.001

import time

start_time = time.time()
for episode in range(num_episodes):
    epsilon = epsilon - 0.001 if epsilon > 0.1 else 0.1
    p1 = 1 - epsilon * (4 / 5)
    p0 = epsilon / 5
    d = {1: p1, 0: p0}
    policy_epsilon = np.vectorize(d.get)(policy)

    init_state = 20  # 初始状态，每个轨迹都从(0,0)->target
    trajectory = gridworld.getTrajectoryScore(nowState=init_state,
                                              action=random.randint(0, 4),
                                              policy=policy_epsilon, steps=trajectorySteps,
                                              stop_when_reach_target=True)
    trajectory.append((17, 4, 1, 17, 4))

    # policy evaluation
    for k in range(len(trajectory)):
        now_state, now_action, now_score, next_state, next_action = trajectory[k]
        expected_qvalue = np.sum(qtable[next_state] * policy_epsilon[next_state])
        TD_target = now_score + gamma * expected_qvalue
        # 用expected_value替代q(next_state, next_action)更好，前者考虑下一步的每个动作，后者只考虑下一步的一个动作
        TD_error = qtable[now_state][now_action] - TD_target
        # TD_error = qtable[now_state][now_action] - (now_score + gamma * qtable[next_state][next_action])
        qtable[now_state][now_action] = qtable[now_state][now_action] - learning_rate * TD_error

    # policy improvement
    policy = np.eye(5)[np.argmax(qtable, axis=1)]
    gridworld.showPolicy(policy)


values = []
for state in range(rows * columns):
    v = 0
    for action in range(5):
        v += policy[state][action] * qtable[state][action]
    values.append(v)
print(np.array(values).reshape(5, -1))
gridworld.show()
print("All state value: ", np.sum(np.array(values)))
end_time = time.time()
print("SARSA-expected time:", end_time - start_time)
