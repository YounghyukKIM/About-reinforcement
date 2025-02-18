# Basic Theory

## Table of Contents
- MDP
- Bellman Equation
- Q-Learning
- SARSA
- Comparison of Q-Learning and SARSA

---

## Preliminary Knowledge

Reinforcement learning is a method of machine learning that takes a different approach from supervised and unsupervised learning. Supervised learning trains on labeled data to solve prediction or classification problems, while unsupervised learning focuses on discovering hidden patterns in data without labels.

In contrast, reinforcement learning is a methodology where an **agent** learns the **optimal actions** based on **rewards** from an **environment**. Unlike supervised learning, reinforcement learning does not require pre-prepared data but **generates data through interactions with the environment**. The **agent** selects an **action** in a given **state** and learns future actions based on received rewards, aiming to **maximize cumulative rewards**.

This method allows learning in complex environments and enables agents to **generate and learn from their own data** through simulations or real-world interactions.

---

- **Monte Carlo**: A method in reinforcement learning that estimates expected rewards based on collected episodes from agent-environment interactions.
- **On-policy**: Learning based on the current policy in use.
- **Off-policy**: Learning where the **target policy** and the **behavior policy** used for data collection are different.
- **Policy (\\( \\pi \\))**: A rule that determines what action the agent should take in a given state.

---

## MDP (Markov Decision Process)

### **Fundamentals of Reinforcement Learning**

MDP is one of the core concepts of reinforcement learning, providing a mathematical framework for an **agent** to interact with an **environment** and determine the **optimal actions (policy)**.

### **Components of MDP**

MDP consists of five main components:

| Component | Description | Example |
|-----------|------------|---------|
| **State (S)** | The set of all possible states the environment can have. | The board configuration in a chess game. |
| **Action (A)** | The set of all possible actions the agent can take in a given state. | Moving a chess piece. |
| **Reward (R)** | The immediate reward received after taking a specific action in a given state. | The reward after moving a piece. |
| **State transition probability (P)** | The probability of transitioning to the next state \\( S' \\) when taking action \\( A \\) in state \\( S \\). | The probability of reaching a particular board configuration after making a move. |
| **Discount Factor (\\( \\gamma \\))** | A factor that determines the importance of future rewards. | |

### **Goal of MDP**

The goal of MDP is for the agent to learn the optimal **policy (\\( \\pi \\))** that maximizes the **cumulative reward (Return)**.

The discount factor \\( \\gamma \\) is used to compute the **return**, which represents the cumulative rewards the agent expects to receive from a certain time step \\( t \\).

#### **Return Formula**
\\[
G_t = \\sum_{k=0}^{\\infty} \\gamma^k R_{t+k}
\\]

A simpler representation:
\\[
G_t = R_t + \\gamma R_{t+1} + \\gamma^2 R_{t+2} + ... + \\gamma^k R_T
\\]

Where:
- \\( R \\) is the immediate reward.
- \\( t \\) represents the time step.
- \\( \\gamma \\) is the discount factor determining how much future rewards matter.

**Example Calculation:**
- \\( Time \\ Step = 10 \\)
- \\( Total \\ Reward = 100 \\)
- \\( \\gamma = 0.9 \\)

Assuming a reward of 10 per step:
\\[
G_t = 10 + (0.9 \\times 10) + (0.9^2 \\times 10) + ... + (0.9^8 \\times 10) = 61.26
\\]

As steps increase, even high rewards decrease in value due to the discount factor \\( \\gamma \\). To ensure learning efficiency, a maximum number of steps per episode is often set.

---

## Bellman Equation

The **Bellman equation** is used to compute the **state-value function** and **action-value function** in reinforcement learning.

### **State-Value Function (\\( V(s) \\))**

The **state-value function** represents the expected cumulative reward for a given state \\( s \\) following a policy \\( \\pi \\).

#### **State-Value Function Formula**
\\[
V^{\\pi}(s) = E_{\\pi}[G_t | S_t = s]
\\]

By recursively expressing return:
\\[
G_t = R_t + \\gamma G_{t+1}
\\]

We derive the Bellman equation:
\\[
V^{\\pi}(s) = E_{\\pi}[R_t | S_t = s] + \\gamma E_{\\pi}[G_{t+1} | S_t = s]
\\]

Expanding further with probabilities:
\\[
V^{\\pi}(s) = \\sum_{a} \\pi(a|s) \\left[ R(s,a) + \\gamma \\sum_{s'} P(s'|s,a) V^{\\pi}(s') \\right]
\\]

### **Action-Value Function (\\( Q(s, a) \\))**

The **action-value function** represents the expected cumulative reward when taking action \\( a \\) in state \\( s \\) and then following policy \\( \\pi \\).

#### **Action-Value Function Formula**
\\[
Q^{\\pi}(s, a) = R(s, a) + \\gamma \\sum_{s'} P(s'|s, a) \\sum_{a'} \\pi(a'|s') Q^{\\pi}(s', a')
\\]

This equation is essential for **Q-Learning**, which approximates optimal Q-values.

---

## Q-Learning

Q-Learning is an **off-policy** algorithm that updates Q-values based on the maximum possible future reward.

#### **Q-Learning Update Formula**
\\[
Q(s,a) \\leftarrow Q(s,a) + \\alpha [R + \\gamma \\max_{a'} Q(s', a') - Q(s,a)]
\\]

- Uses **greedy updates** to maximize future reward estimates.
- Behaves **off-policy** since it updates using the best possible action, not necessarily the action taken.

---

## SARSA

SARSA is an **on-policy** algorithm that updates Q-values based on the actual action taken according to the behavior policy.

#### **SARSA Update Formula**
\\[
Q(s,a) \\leftarrow Q(s,a) + \\alpha [R + \\gamma Q(s', a') - Q(s,a)]
\\]

- Uses the next **actual action** taken instead of the greedy best action.
- Behaves **on-policy** as it learns from the actions chosen by the behavior policy.

---

## Comparison of Q-Learning and SARSA

| Algorithm  | Policy Type | Action Selection | Behavior |
|------------|------------|------------------|----------|
| **Q-Learning** | Off-policy | Uses \\( \\max Q(s', a') \\) | More exploratory, finds optimal policy faster |
| **SARSA** | On-policy | Uses \\( Q(s', a') \\) (actual next action) | More conservative, safer learning |

Q-Learning generally learns faster but can be unstable, while SARSA follows a safer, more gradual approach.

---
