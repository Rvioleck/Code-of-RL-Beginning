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

epsilon = 0.5
p1 = 1 - epsilon * (4 / 5)  # epsilon=1时，p1=p2=0.2，此时为随机策略，全力进行探索，收集可能存在的更优策略
p0 = epsilon / 5  # epsilon=0时，p1=1,p2=0，此时为贪婪策略，大体策略已经固定，尽力优化已知的策略
d = {1: p1, 0: p0}
behavior_policy = np.vectorize(d.get)(policy)  # 随机策略

gridworld.show()  # 打印gridworld
gridworld.showPolicy(policy)

trajectorySteps = 20000

num_episodes = 600
learning_rate = 0.001

import time

start_time = time.time()
for episode in range(num_episodes):
    init_state = 0  # 初始状态，每个轨迹都从init_state->target
    trajectory = gridworld.getTrajectoryScore(nowState=init_state,  # 收集轨迹数据，起点为init_state，终点为预设end_state
                                              action=random.randint(0, 4),
                                              policy=behavior_policy,  # 探索时用behavior_policy
                                              steps=trajectorySteps,
                                              stop_when_reach_target=True)
    # 由于每个trajectory都是由init_state->end_state，因此这条轨迹的学习结果最好，其它state->end_state的学习策略不一定最优
    trajectory.append((17, 4, 1, 17, 4))

    # policy evaluation
    for k in range(len(trajectory)):  # 正反遍历都是一样的，因为没算acc_reward
        now_state, now_action, now_score, next_state = trajectory[k][0:4]
        # 因为理论上此处expected_qvalue仍然依赖于当前策略，所以不能用behaviour_policy算出来的的结果作为TD-target
        expected_qvalue = np.sum(qtable[next_state] * policy[next_state])
        # TD-target由now_score + gamma * qtable[next_state][next_action]变为了now_score + gamma * np.max(qtable[next_state])
        TD_target = now_score + gamma * expected_qvalue
        TD_error = qtable[now_state][now_action] - TD_target
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

print("All state value: ", np.sum(np.array(values)))
end_time = time.time()
print("Q-learning off-policy time:", end_time - start_time)
