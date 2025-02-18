# Basic theory

Table of Contents
- MDP
- Bellman Equation
- Q-Learning
- SARSA
- Q-Learning과 SARSA 비교
---

## Basic Knowledge Before Starting

Reinforcement learning is a method of machine learning that takes a different approach from supervised and unsupervised learning. 

Supervised learning trains on labeled data to solve prediction or classification problems, while unsupervised learning focuses on uncovering hidden patterns in data without labels.

In contrast, reinforcement learning learns the **optimal actions** based on **rewards** in a given **environment**. Unlike other learning methods, reinforcement learning does not require pre-prepared data; instead, it **generates data through interactions** with the **environment**.

An **agent** selects an **action** in a given **state** and learns from the resulting reward, aiming to **maximize cumulative rewards** over time.

This approach allows learning even in complex environments and enables agents to **generate their own training data** through simulations or real-world interactions.

---
- **Monte Carlo**: A method in reinforcement learning that estimates the expected reward based on episodes collected through the agent's interaction with the environment.
- **On-policy**: Learning is based on the policy currently being used during training.
- **Off-policy**: The **target policy** used for learning differs from the **behavior policy** used for data collection.
- **$\pi$ (Policy)**: A rule that determines which action an agent should take in a given state.
---

## MDP(Markov Decision Process)

**Basic Concepts of Reinforcement Learning**

One of the key concepts in reinforcement learning is the **mathematical framework** used by an **agent** to determine the **optimal policy** while interacting with the **environment**.

### **Components of an MDP**
A **Markov Decision Process (MDP)** consists of **five components**:

| Component                     | Description                                                         | Example                                   |
|--------------------------------|---------------------------------------------------------------------|-------------------------------------------|
| **State (S)**                 | The set of all possible states the environment can have.           | The board configuration in a chess game. |
| **Action (A)**                | The set of all possible actions the agent can take in each state.  | Moving a chess piece.                     |
| **Reward (R)**                | The immediate reward received for taking a specific action.        | The reward assigned after moving a piece. |
| **State Transition Probability (P)** | The probability of transitioning to the next state \( S' \) when taking action \( A \) in the current state \( S \). | The probability of a piece landing in a specific position after a move. |
| **Discount Factor (γ)**        | Determines the importance of future rewards.                      | —                                         |


These components form the foundation of reinforcement learning.

## **Goal of MDP (Markov Decision Process)**

### **What is the goal of MDP?**
The goal of an MDP is for an **agent** to learn the **optimal policy $(\pi)$** that maximizes the **cumulative reward (Return)**.

Among the components discussed earlier, the **discount factor $(\gamma)$** might not be immediately clear. The discount factor is used in MDP to compute **Return**.


### **What is Return?**
Return represents the cumulative reward an agent will receive **after time step $(t)$**. Future rewards are **weighted with the discount factor $(\gamma)$**, making them **less significant over time**. This design ensures that **immediate rewards are prioritized over distant future rewards**.


### **Return Formula**

$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k}$

Rewriting this in a more intuitive way:

$G_t = R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \dots + \gamma^k R_T$


### **Explanation of the Formula**
- $( R )$: The immediate reward received at each time step.
- $( t )$: Represents the time step.
- $( \gamma )$ (discount factor): Determines how much future rewards should be valued.


### **Example Calculation**
Assume the following:
- **Time steps**: 10
- **Total reward**: 100
- **\(\gamma\)**: 0.9
- **Reward per step**: 10

Computing return:

$G_t = 10 + (0.9 \times 10) + (0.9^2 \times 10) + (0.9^3 \times 10) + \dots + (0.9^8 \times 10)$

$G_t = 61.2579511$

As the time steps increase, even if good rewards are obtained, **their contribution to the total return decreases** due to the discount factor.


### **Why Limit the Number of Steps?**
In the formula, we assumed an **infinite** number of steps. However, in practice:
- **Too many steps** can make learning inefficient or unstable.
- Therefore, we **limit the maximum number of steps per episode** to ensure effective training.

### **Conclusion**
In an MDP, the objective is to **maximize cumulative reward within a finite or constrained number of steps**.

---
## **Bellman Equation**

### **What is the Bellman Equation?**
The **Bellman equation** is used to compute the **state-value function** or **action-value function** in reinforcement learning. It utilizes the concepts of **MDP (Markov Decision Process)** to derive these functions.

---

### **State-Value Function**
#### **What is the state-value function?**
The **state-value function** represents the **expected cumulative reward** an agent will obtain **when starting from a specific state** and following a given policy.

#### **State-Value Function Formula**
$V^{\pi }(s)=E_{\pi}[G_{t}|S_{t}=s]$

#### **Explanation of the Formula**
- $V^{\pi}(s)$ : Expected cumulative reward when following policy $\pi$ from state $s$.
- $G_{t}$ : Return (cumulative reward), as defined in MDP.
- $S_{t} = s$ : The state $s$ at time step $t$.
- $|$ : Conditional expectation (given the state $s$ at time $t$).
- $\mathbb{E}[G_{t} | S_{t} = s]$ : Expected cumulative reward starting from state $s$.

While this is a general representation, the **Bellman equation** refines this concept by incorporating MDP components.

---

### **Derivation of the Bellman Equation**
#### **Starting with the State-Value Definition**

$V^{\pi }(s)=E_{\pi}[G_{t}|S_{t}=s]$

#### **Using the Return Formula**

From our previous discussion on **Return**:

$G_t = \sum_{k=0}^{\infty}\gamma ^{k}R_{r+k}$

Rewriting it in a recursive form:

$\rightarrow$ $G_{t} = R_{t}+\gamma G_{t+1}$

#### **Applying the Recursive Return to the State-Value Function**

$\rightarrow$ $V^{\pi }(s)=E_{\pi}[R_{t}+\gamma G_{t+1}|S_{t}=s]$

Using the **linearity of expectation**:

$\rightarrow$ $V^{\pi }(s)=E_{\pi}[R_{t}|S_{t}=s]+\gamma E_{\pi}[G_{t+1}|S_{t}=s ]$

#### **Expanding the Expectation Terms**
The expected immediate reward can be rewritten as:

$\rightarrow$ $E_{\pi}[R_{t}|S_{t}=s]$ = $\sum_{a}^{}\pi(a|s)R(s,a)$

Similarly, the expectation of the future value function:

$\rightarrow$ $E_{\pi}[G_{t+1}|S_{t}=s ]$ = $\sum_{a}^{}\pi(a|s)\sum_{s'}^{}P(s'|s,a)V^{\pi}(s)$

#### **Final Bellman Equation for State-Value Function**

$V^{\pi }(s) = \sum_{a}^{}\pi(a|s)R(s,a)\sum_{s'}^{}P(s'|s,a)V^{\pi}(s)$

This equation expresses the **recursive relationship** between the value of a state and the values of future states, weighted by the probability of transitioning to those states.

## **Action-Value Function**

### **What is the Action-Value Function?**
The **action-value function** represents the **expected cumulative reward** an agent will obtain when **taking a specific action** in a given state and following a policy afterward.

#### **Action-Value Function Formula**
$Qπ(s)=Eπ​[Gt​∣St​=1s,A_{t}=a]$

### **Derivation of the Action-Value Function**
Rewriting the equation in a probabilistic expectation form:

$\rightarrow$ $Q^\pi(s,a)=R(s,a)+\gamma \sum_{s'}P(s'|s,a)V^{\pi}(s')$

Since $V^{\pi}(s')$ can be expanded using policy $\pi$:

$\rightarrow$ $V^{\pi}(s')=\sum_{a'}\pi(a'|s')Q^{\pi}(s',a')$

Substituting $V^{\pi}(s')$ into the action-value function equation:
$Q^{\pi}(s,a) = R(s,a) + \gamma \sum_{s'} P(s'|s,a) \sum_{a'} \pi(a'|s') Q^{\pi}(s',a')$

Thus, the **action-value function** is derived by expanding the **state-value function** with the **probability distribution of actions** under the policy.

---

## **Why Did We Derive These Equations?**

In **reinforcement learning** and **deep reinforcement learning**, we do not directly use the **state-value function** or **action-value function** in their raw forms. Instead, we utilize a modified version of the action-value function known as **Q-learning**.

At this point, you might ask:
**"If we do not use the state-value and action-value functions directly, why did we derive them?"**

The reason for deriving these equations is to **understand the theoretical foundation** of **Q-learning**, which will be discussed later.

### **The Connection to Q-learning**
Q-learning is fundamentally based on the **action-value function** and works by approximating **Q-values** through interaction with the environment.

Although Q-learning refines **Q-values** over time, its foundation still originates from the **Bellman equation**. Understanding the Bellman equation allows us to grasp **why Q-learning works and how it updates values efficiently**.

### **Why Not Use the Action-Value and State-Value Functions Directly?**
- **State-value and action-value functions provide optimal solutions** by evaluating all possible states and actions.
- In **small environments**, they can be **computed efficiently** with minimal computational cost.
- However, **modern reinforcement learning problems** involve **large and complex environments** with **an enormous number of states and variables**.
- **Direct computation of all state-values and action-values becomes computationally infeasible** as the complexity of the environment increases.

### **How Does Q-learning Solve This Problem?**
To address the computational limitations, **Q-learning**:
- **Avoids computing all possible values upfront**.
- **Learns optimal action-values through direct interaction with the environment**.
- **Gradually approximates Q-values instead of storing and computing them explicitly**.

By experiencing the environment iteratively, **Q-learning finds an optimal policy without requiring exhaustive computation**.

### **Conclusion**
The derivation of state-value and action-value functions helps **build an intuition** for **how Q-learning approximates optimal solutions**. Understanding these concepts allows us to fully appreciate the **efficiency and effectiveness** of modern reinforcement learning techniques.

---

### **Q-learning**
Q-learning is a method that learns by directly interacting with the environment and updating Q-values based on received rewards. It is a representative **Off-Policy** reinforcement learning algorithm.

**Q-learning Update Formula**
- $Q(s,a) \leftarrow Q(s,a)+\alpha [R+\gamma \max_{a'} Q(s', a')-Q(s,a)]$

This equation updates the Q-value based on the reward obtained from taking a specific action in the current state and the maximum Q-value of the next state.

### **Step-wise Update and Difference from Monte Carlo**
Q-learning updates values at **each step**. That is, it updates Q-values immediately after taking an action. In contrast, Monte Carlo methods update Q-values **only after the episode ends**, using the total accumulated rewards.

### **Off-Policy and $\varepsilon$-Greedy Policy**
Q-learning is **Off-Policy** because the **target policy** used for learning and the **behavior policy** used for data collection are different.

The behavior policy in Q-learning is typically an **$\varepsilon$-greedy policy**. The parameter $\varepsilon$ (between 0 and 1) determines whether the agent **explores** randomly or **exploits** the Q-values to make a decision.

### **Example**
- **Current state**: $(s=0)$
- **Possible action space** = $[0,1,2]$
- **Q-values**:
  - $Q(0, 0) = 0, \quad Q(0, 1) = 1, \quad Q(0, 2) = 2$
- **$\varepsilon = 0.5$**

In this case:
- There is a **50% probability** that the agent selects $a = 2$ based on the Q-value ($Q(0, 2)$).
- There is a **50% probability** that the agent selects an action randomly from $[0,1,2]$.

### **Update Process**
Assume the following:
- **State**: $(s=0)$
- **Action**: $(a=2)$ # Selected action
- **Reward**: $(r=5)$
- **Next state**: $(s'=1)$
- **Possible action space** = $[0,1,2]$
- **Q-values**:
  - $Q(1,0) = 2, \quad Q(1,1) = 3, \quad Q(1,2) = 4$
- Assume that the agent selected an action randomly with a 50% $\varepsilon$ probability.

**Q-learning Update Formula**:
$Q(s,a) \leftarrow Q(s,a)+\alpha [R+\gamma \max_{a'} Q(s', a')-Q(s,a)]$

**Hyperparameters**:
- Learning rate ($\alpha$): 0.1
- Discount factor ($\gamma$): 0.9

**Update Calculation**:
- **Current Q-value**: $Q(s=0,a=2) = 2$
- **Maximum Q-value in next state**:
  - $\max_{a'} Q(s'=1,a') = \max\{Q(1,0),Q(1,1),Q(1,2)\} = 4$ # Because $Q(1,2)$ is the largest.

Applying the update equation:

$(s=0,a=2) \leftarrow 2 + 0.1 [5 + 0.9 \cdot 4 - 2]$

**Calculation**:

$\rightarrow Q(s=0,a=2) \leftarrow 2 + 0.1 [5 + 3.6]$

$\rightarrow Q(s=0,a=2) \leftarrow 2 + 0.1 \cdot 6.6$

$\rightarrow Q(s=0,a=2) \leftarrow 2 + 0.66 = 2.66$

**Final Result**:

$\rightarrow Q(s=0,a=2) = 2.66$

During exploration, the agent follows the **behavior policy ($\varepsilon$-greedy policy)**, but during learning, it always updates based on the **optimal action** in the next state ($\max_{a'} Q(s', a')$). Since the behavior policy and the target policy are different, **Q-learning is an Off-Policy method**.

### **SARSA**
SARSA is a representative **On-Policy** reinforcement learning algorithm. The agent interacts with the environment, selects actions, and updates values accordingly. SARSA stands for **"State-Action-Reward-State-Action"**, meaning it updates Q-values based on the current state, action, reward, next state, and next action.

**SARSA Update Formula**
- $Q(s,a) \leftarrow Q(s,a)+\alpha[R+\gamma Q(s',a')-Q(s,a)]$

This equation updates the Q-value based on the **actual action taken** in the next state, meaning the behavior policy **directly affects the learning process**.

### **On-Policy and $\varepsilon$-Greedy Policy**
SARSA is an **On-Policy** algorithm because the **target policy** used for learning and the **behavior policy** used for data collection are the same.

Like Q-learning, SARSA typically uses an **$\varepsilon$-greedy policy**, where $\varepsilon$ determines the balance between **exploration (random action selection)** and **exploitation (choosing the best action based on Q-values)**.

### **Example**
Using the same assumptions as before:
- **State**: $(s=0)$
- **Action**: $(a=2)$ # Selected action
- **Reward**: $(r=5)$
- **Next state**: $(s'=1)$
- **Possible action space** = $[0,1,2]$
- **Q-values**:
  - $Q(1,0) = 2, \quad Q(1,1) = 3, \quad Q(1,2) = 4$
- Assume that a **random action was chosen with 50% probability**.

**Hyperparameters**:
- Learning rate ($\alpha$): 0.1
- Discount factor ($\gamma$): 0.9

**Update Process**:
- **Current state**: $s = 0$
- **Action**: $a = 2$ (selected action)
- **Reward**: $r = 5$
- **Next state**: $s' = 1$
- **Next action**: $a' = 1$
- **Q-values**:
  - $Q(1,0) = 2, \quad Q(1,1) = 3, \quad Q(1,2) = 4$

**Update Calculation**:
- **Current Q-value**:
  - $Q(s=0,a=2) = 2$
- **Q-value of next action in next state**:
  - $Q(s'=1,a'=1) = 3$

Applying the update equation:

$Q(s=0,a=2)\leftarrow2+0.1[5+0.9\cdot 3-2]$

**Calculation**:
- $Q(s=0,a=2)\leftarrow2+0.1[5+2.7-2]$
- $Q(s=0,a=2)\leftarrow2+0.1\cdot 5.7$
- $Q(s=0,a=2)\leftarrow2+0.57$

**Final Result**:

$Q(s=0,a=2) = 2.57$

Since **SARSA updates based on the actual action taken**, the behavior policy and the target policy are identical, making SARSA an **On-Policy** learning method.
