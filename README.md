# 强化学习课程项目

## 项目一

项目一采用Value Iteration的方法，在每个状态价值收敛后根据价值更新策略，直到策略收敛。

在 0.23.1版本的gym中FrozenLake-v0已停用，需使用FrozenLake-v1：

```python
env = gym.make('FrozenLake-v1')
```

## 项目二

项目二是用蒙特卡洛和时间差分对价值函数分别进行评估。在动作选取的时候采用epsilon-greedy的方式。在两种方法中epsilon的衰减方式有差别。对训练的eipsode数量即代码中的`num_episodes`变量对胜率的影响变化图如下：

### MC

### ![](.\MC.png)

### TD

![](.\TD.png)

### 项目三

在训练过程中reward的变化图如下所示

### Q-Learning

![Q-Learning](.\Q-Learning.png)

### Sarsa

![](.\Sarsa.png)

将向上和向右的reward都设为0，可实现走更安全的路。

未修改前Q-Learning路径：[36, 24, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 35]

修改后的Q-Learning路径：[36, 24, 12, 0, 1, 2, 3, 4, 5, 6, 18, 19, 20, 21, 22, 34, 35]

修改后的路径更为安全。