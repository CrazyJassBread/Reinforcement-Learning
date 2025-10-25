# Reinforcement Learning
最近正在学习CS234，可是课程主要侧重算法而非代码实践，于是对涉及到的经典算法都找了合适的env手动进行实现

代码部分主要依赖于[动手学强化学习](https://hrl.boyuai.com/)的内容，但是官方给出的代码基于gym库，一些接口与现在的gymnasium并不适配，所以对一些函数的接口进行了简单的修改

目录；
- sarsa_vs_q_learning：这个文件夹中用来展示 sarsa 算法和 qlearning 的区别
- Function_approximation：使用函数近似表示V或Q（应对连续状态情况）
- Double-DQN VS DQN：实现了DDQN和DQN并且进行了比较
- Policy Gradient：策略梯度算法，目前实现了REINFORCE和PPO