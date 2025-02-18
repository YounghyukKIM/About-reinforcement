# 기초이론

목차
- MDP
- Bellman Equation
- Q-Learning
- SARSA
- Q-Learning과 SARSA 비교
---
시작하기전 기본 지식

강화학습은 머신러닝의 한 방법으로, 지도학습과 비지도학습과는 다른 접근 방식을 취합니다. 지도학습은 레이블이 있는 데이터를 학습하여 예측하거나 분류 문제를 해결하고, 비지도학습은 레이블 없이 데이터의 숨겨진 패턴을 찾는 데 초점을 맞춥니다.

반면, 강화학습은 어떠한 **환경**에서 **보상(Reward)** 을 기반으로 **최적의 행동(Action)** 을 학습하는 방법론입니다. 강화학습은 데이터를 미리 준비할 필요가 없으며, **환경(Environment)** 과의 **상호작용**을 통해 **데이터를 생성**합니다. **에이전트(Agent)** 는 **상태(State)** 에서 **행동(Action)** 을 선택하고, 그 결과로 얻은 보상을 바탕으로 미래의 행동을 학습하여 **누적 보상을 최대화**하는 방향으로 나아갑니다.

이 방법론은 복잡한 환경에서도 학습을 가능하게 하며, 시뮬레이션이나 실세계의 환경을 활용하여 **에이전트가 스스로 데이터를 만들어 학습**할 수 있습니다.

---
- Monte Carlo : 강화학습에서 에이전트가 환경과 상호작용하면서 수집한 에피소드를 기반으로 기대 보상을 추정하는 기법
- On policy : 학습 중 현재 사용 중인 정책을 기반으로 학습.
- Off policy : 학습 중 **목표 정책(target policy)** 과 데이터 수집에 사용하는 **행동 정책(behavior policy)** 이 다름.
- $\pi$(정책) : 정책은 에이전트가 특정 상태에서 어떤 행동을 선택할지 결정하는 규칙

---

## MDP(Markov Decision Process)

**강화학습 기본 개념**

강화학습의 핵심 개념 중 하나로, **에이전트(Agent)** 가 **환경(Environment)** 과 상호작용하면서 **최적의 행동(Policy)** 을 결정하기 위해 사용하는 수학적 프레임워크입니다.

**MDP의 구성요소**로는 총 **5가지**가 있습니다.
> |구성|설명|예|
|------|---|---|
|State : 상태 / S|환경이 가질 수 있는 모든 상태의 집합.|체스 게임에서 각 말의 배치 상태.|
|Action : 행동 / A|에이전트가 각 상태에서 선택할 수 있는 모든 행동의 집합.|체스 말의 이동.|
|Reward : 보상 / R|특정 상태에서 특정 행동을 취했을 때 받는 즉각적인 보상.|체스 말의 이동한 후 발생하는 보상|
|State transition probability : 전이확률 / P|에이전트가 현재 상태 S에서 행동A를 취하였을 때 다음 상태S'이 될 확률|현재 말의 상태에서 특정 행동을 선택했을 때 다음 가능한 위치로 이동할 확률|
|Discount Factor : 할인률 / γ|미래 보상의 중요도를 결정하는 값||

위 와 같은 구성으로 있습니다.

---

### MDP(Markov Decision Process) 목표

**MDP의 목표는?**
MDP의 목표는 에이전트가 최적의 **정책(Policy, π)** 을 학습하여 **누적 보상(Return)** 을 최대화하는 것입니다.

위에서 다루었던 구성요소 중 $\gamma$가 이해가 되지 않았을 수도 있습니다 할인률은 바로 MDP의 목표에서 Retrun을 계산하기 위해서 사용됩니다.

**Retrun 이란?**
에이전트가 시점 t이후에 받을 누적 보상을 나타냅니다. 미래의 보상은 할인인자 (γ)를 통해 가중치를 감소시키며, 이는 보상이 시간에 따라 점차 덜 중요하게 취급되도록 설계된 것입니다.

**Return 공식**
$\rightarrow$ $G_t = \sum_{k=0}^{\infty}\gamma ^{k}R_{r+k}$
이러한 형태의 수식이며 조금더 쉽게 작성해보면

$\rightarrow$ $G_t = R_t+\gamma R_{t+1}+\gamma^{2} R_{r+2} + ... + \gamma^{k}R_{T}$
이러한 형태의 수식으로 만들 수 있습니다.

**수식 설명**
이 수식을 설명해 보자면
$R$ : 은 보상으로 즉각적인 형태에서 받는 보상입니다.
$t$ : 각 타임스텝의 시간을 의미합니다.
$\gamma$ : 감가율로 미래보상을 얼마나 중요시 여길지 결정하는 값입니다.


**예제**

가정
$Time step$ = 10
$total Reward$ = 100
$\gamma$ = 0.9
각 Step 마다 받은 보상 10이라고 가정, 


$G_{t}=10+(0.9*10)+(0.9^{2}*10)+(0.9^{3}*10)+(0.9^{4}*10)+(0.9^{5}*10)+(0.9^{6}*10)+(0.9^{7}*10)+(0.9^{8}*10)=61.2579511$

이러한 식으로 스텝이 길어질 수록 좋은 보상을 받는다고 하더라도 보상을 줄어듭니다.

위 수식에서는 스텝이 무한하다는 가정을 했었지만 보통 스텝이 너무 길어지면 학습이 제대로 되지 않을 가능성이 높기에 한 에피소드에 가능한 최대 스텝을 제한하기도 합니다.

즉 MDP에서는 제한되거나 유한한 Step 속에서 최대의 누적보상을 얻고자 하는 것 입니다.

---

## Bellman Equation

**Bellman Equation이란?**
Bellman equation은 보통 상태가치(State Value)함수 또는 행동가치(Action Value)함수를 구하기 위해서 사용됩니다.

Bellman equation은 위와같은 함수를 구하기 위해서 MDP의 개념들이 사용됩니다.

### 상태가치함수
**상태가치함수란?**

**특정 상태(state)**에 있을 때, 미래에 얻을 것으로 예상되는 누적 보상의 **기대값**입니다.

**상태가치함수 공식**

$\rightarrow$ $V^{\pi }(s)=E_{\pi}[G_{t}|S_{t}=s]$

**수식 설명**
$V^{\pi}(s)$ : 특정 상태 $s$에서 정책$\pi$를 따랐을 때 기대되는 누적보상
$G_{t}$ : MDP에서 다루었던 누적보상(Return)
$S_{t}=s$ : $t$시점의 값$s$
$|$ = 조건부 기댓값
$E[G_{t}|S_{t}=s]$ = 시점$t$에서 상태 값$s$에서 얻을 수 있는 누적보상 기댓값의 평균

상태가치를 한번 하나씩 설명을 해보았습니다. 그러나 Bellman equation에서의 상태가치는 이렇게 단순한 형태로 사용되지 않습니다. 위에서 다루었던 MDP와 연계하여 사용합니다.

**수식 유도**

기존 상태가치
$V^{\pi }(s)=E_{\pi}[G_{t}|S_{t}=s]$

우리가 알고 있던 리턴값의 공식

$G_t = \sum_{k=0}^{\infty}\gamma ^{k}R_{r+k}$

재귀적 형태로 변경

$\rightarrow$ $G_{t} = R_{t}+\gamma G_{t+1}$

재귀적 Return 형태의 상태가치

$\rightarrow$ $V^{\pi }(s)=E_{\pi}[R_{t}+\gamma G_{t+1}|S_{t}=s]$

즉시보상을 보현하기 위해서 기댓값의 선형성 특징을 유도

$\rightarrow$ $V^{\pi }(s)=E_{\pi}[R_{t}|S_{t}=s]+\gamma E_{\pi}[G_{t+1}|S_{t}=s ]$

또한 $E_{\pi}[R_{t}|S_{t}=s]$은 $\sum_{a}^{}\pi(a|s)R(s,a)$이렇게 표현 할 수 있고, $E_{\pi}[G_{t+1}|S_{t}=s ]$은 $\sum_{a}^{}\pi(a|s)\sum_{s'}^{}P(s'|s,a)V^{\pi}(s)$ 이렇게 표할 수 있을 것 입니다.

그럼 최종적인 유도는 이러한 형태로 유도할 수 있을 것 입니다

$\rightarrow$ $V^{\pi }(s) = \sum_{a}^{}\pi(a|s)R(s,a)\sum_{s'}^{}P(s'|s,a)V^{\pi}(s)$

---

### 액션가치(Action Value)함수

**특정 상태(state)**에서 **특정 행동(action)**을 취하였을 때, 미래에 얻을 것으로 예상되는 누적 보상의 **기대값**입니다.

**액션가치함수 공식**
$Qπ(s)=Eπ​[Gt​∣St​=1s,A_{t}=a]$


**수식 유도**
위 공식을 기대값의 확률적 표현으로 봐꾸어 보면

$\rightarrow$ $Q^\pi(s,a)=R(s,a)+\gamma \sum_{s'}P(s'|s,a)V^{\pi}(s')$
이러한 형태로 바꿀 수 있으며

$V^{\pi}(s')$를 정책 $\pi$를 통해 아래와 같이 확장할 수 있습니다.

$\rightarrow$ $V^{\pi}(s')=\sum_{a'}\pi(a'|s')Q^{\pi}(s',a')$

이 $V^{\pi}(s')$를 $Q^{\pi}(s,a)$에 대입하면 아래와 같이 표현 할 수 있을 것 입니다.

$\rightarrow$ $Q^{\pi}(s,a)=R(s,a)+\gamma \sum_{s'}P(s'|s,a) \sum_{a'}\pi(a'|s')Q^{\pi}(s'|s')$


즉 액션가치 함수는 상태가치에 액션에 대한 확률값들을 그대로 유도하여 더하면 되는 공식입니다.

**왜 유도했는가?**

사실 Reinforcement learning이나 Deep Reinforcement learning에서는 이러한 액션가치나 상태가치를 사용하지 않고 **액션가치를 변형한 형태**인 **Q-learning**를 사용합니다.

그럼 질문이 들어오겠죠.
"액션가치와 상태가치를 그대로 사용하지 않음에도 불구하고 왜 유도를 하였나요?"

액션 가치와 상태 가치를 유도한 이유는, 이후 다룰 Q-learning의 이론적 기반을 이해하기 위함입니다.

Q-learning은 기본적으로 액션 가치 함수에 기반하여 동작하며, 환경을 경험하며 Q-value를 근사화하는 방법입니다.

Q-learning의 내용은 아래에서 다루겠지만 환경을 경험하며 Q-Value를 근사화 하는 방법입니다.

액션 가치 함수와 상태 가치 함수는 환경 내 모든 상태에서 최적의 방향을 결정할 수 있는 정답을 제공합니다. 이는 작은 환경에서는 계산 비용이 적기 때문에 실질적으로 사용할 수 있습니다. 하지만 최근 연구자들이 다루는 환경은 매우 복잡하고 크며, 모든 상태와 변수에 대해 계산해야 하므로 계산 비용이 기하급수적으로 증가합니다.

따라서 Q-learning은 이러한 복잡한 환경에서 계산 비용을 줄이기 위해, 환경을 경험하며 근사적으로 답을 찾는 효율적인 방법으로 설계되었습니다. 이러한 개념을 이해하기 위해 먼저 상태 가치와 액션 가치의 수식을 유도했습니다.

---

### Q-learning
Q-Learning은 환경을 직접 경험하며 보상을 기반으로 Q-값을 업데이트하고 학습하는 방법입니다. Q-Learning은 강화학습의 대표적인 Off-Policy 알고리즘입니다.

**Q-Learning의 업데이트 수식**
$Q(s,a) \leftarrow Q(s,a)+\alpha [R+\gamma \max_{a'} Q(s', a')-Q(s,a)]$

이 수식은 현재 상태에서 특정 행동을 취한 결과로 얻은 보상과, 다음 상태에서 최적 행동을 선택했을 때의 Q-값을 기반으로 값을 업데이트합니다.

**Step 업데이트와 Monte Carlo와의 차이**

Q-Learning은 Step마다 값을 업데이트하는 방식입니다. 즉, 행동 후 즉시 Q-값을 업데이트하며 학습을 진행합니다. 반면, Monte Carlo는 에피소드가 끝난 후 모든 누적 보상을 계산해 한 번에 Q-값을 업데이트합니다.


**Off-Policy와 $\varepsilon$-탐욕적 정책**
Q-Learning이 Off-Policy인 이유는 학습 중 사용하는 **목표 정책(target policy)** 과 데이터를 수집하는 **행동 정책(behavior policy)** 이 다르기 때문입니다.

Q-Learning에서 행동 정책으로는 주로 $\varepsilon$-탐욕적 정책을 사용합니다. $\varepsilon$은 0과 1 사이의 확률값으로, 에이전트가 다음 행동을 결정할 때 무작위 탐험(Exploration)을 할지, Q-값을 기반으로 행동을 선택할지(Exploitation)를 결정합니다.

**예제**
- 현재 상태 : $(s=0)$
- 가능한 Action Space = $[0,1,2]$
- Q-Value
	- $Q(0, 0) = 0, \quad Q(0, 1) = 1, \quad Q(0, 2) = 2$
- $\varepsilon = 0.5$

이 경우
- 50% 확률로 \(Q(0, 2)\) 값에 따라 행동 \(a = 2\)를 선택합니다.
- 50% 확률로 Action Space \([0, 1, 2]\)에서 무작위로 행동을 선택합니다.

**업데이트 과정**
위 가정을 그대로 사용.
- 상태 : $(s=0)$
- 행동 : $(a=2)$ #선택한 행동
- 보상 : $(r=5)$
- 다음 상태 : $(s'=1)$
- 가능한 Action Space = $[0,1,2]$
- Q-Value = $Q(1,0)=2,Q(1,1)=3,Q(1,2)=4$
- 50%의 입실론 정책으로 50%확률로 랜덤한 액션을 선택하였다고 가정

- Q-learning update 공식 : $Q(s,a) \leftarrow Q(s,a)+\alpha [R+\gamma \max_{a'} Q(s', a')-Q(s,a)]$

하이퍼 파라미터
- learning rate($\alpha$) : 0.1
- gamma($\gamma$) = 0.9

업데이트 과정
- 현재 $Q(s=0,a=2) = 2$
- 다음 상태 s' = 1에서 최대 Q-value 계산
	$max_{a'}Q(s'=1,a') = max{Q(1,0),Q(1,1),Q(1,2)} = 4$ # Q(1,2)의 값이 가장 크기 때문.
    
업데이트 공식 대입
$(s=0,a=2) \leftarrow 2+01[5+0.9 \cdot4-2]$

계산
$\rightarrow Q(s=0,a=2) \leftarrow 2+0.1[5+3.6]$
$\rightarrow Q(s=0,a=2) \leftarrow 2+0.1 \cdot 6.6$
$\rightarrow Q(s=0,a=2) \leftarrow 2+0.66 = 2.66$

결과
$\rightarrow Q(s=0,a=2) = 2.66$


탐험 시 에이전트는 행동 정책($\varepsilon$-탐욕적 정책)을 사용하지만, 학습 시에는 항상 다음 상태 $(s')$에서 최적 행동을 선택하는 $(\max_{a'} Q(s', a'))$를 기반으로 학습합니다. 이처럼 행동 정책과 목표 정책이 분리되어 있기 때문에 Q-Learning은 Off-Policy 방법입니다.

### SARSA
Sarsa는 강화학습의 대표적인 On-Policy 알고리즘으로, 에이전트가 환경과 상호작용하며 행동을 선택하고 학습하는 방법입니다. Sarsa는 "State-Action-Reward-State-Action"의 약자로, 현재 상태와 행동, 보상, 다음 상태와 행동을 기반으로 학습이 진행됩니다.

**SARSA의 업데이트 공식**
$Q(s,a) \leftarrow Q(s,a)+\alpha[R+\gamma Q(s',a')-Q(s,a)]$
이 수식은 다음 상태에서 실제로 수행한 행동의 Q-값을 기반으로 값을 업데이트합니다.
즉, 행동 정책이 학습 과정에서 직접적으로 영향을 미칩니다.

On-Policy와 $\varepsilon$-탐욕적 정책
Sarsa는 On-Policy 알고리즘으로, 학습 중 사용하는 **목표 정책(target policy)** 과 데이터를 수집하는 **행동 정책(behavior policy)** 이 동일합니다.

행동 정책으로 주로 $\varepsilon$-탐욕적 정책을 사용합니다.

$\varepsilon$: 에이전트가 무작위 탐험(Exploration)을 할지, Q-값을 기반으로 행동을 선택할지(Exploitation)를 결정.

**예제**
위 가정을 그대로 사용.
- 상태 : $(s=0)$
- 행동 : $(a=2)$ #선택한 행동
- 보상 : $(r=5)$
- 다음 상태 : $(s'=1)$
- 가능한 Action Space = $[0,1,2]$
- Q-Value = $Q(1,0)=2,Q(1,1)=3,Q(1,2)=4$
- 50%의 입실론 정책으로 50%확률로 랜덤한 액션을 선택하였다고 가정

하이퍼 파라미터
- learning rate($\alpha$) : 0.1
- gamma($\gamma$) = 0.9

업데이트 과정
- 현재 상태 : $s = 0$
- 행동 : $a = 2$(선택한 행동)
- 보상 : $r = 5$
- 다음 상태 : %s' = 1%\
- 다음 행동 : %a' = 1%
- Q-Value:
	- (1,0)=2,Q(1,1)=3,Q(1,2)=4$

계산
- 현재 Q값
	- Q(s=0,a=2) = 2
- 다음 상태의 s'=1에서 다음 행동 a'= 1의 Q값 : Q(s'= 1,a'=1)=3

업데이트 공식 대입
$Q(s=0,a=2)\leftarrow2+0.1[5+0.9\cdot 3-2]$
$Q(s=0,a=2)\leftarrow2+0.1[5+2.7-2]$
$Q(s=0,a=2)\leftarrow2+0.1\cdot 5.7$
$Q(s=0,a=2)\leftarrow2+0.57$

결과
$Q(s=0,a=2) = 2.57$
Sarsa는 이처럼 실제로 선택한 행동의 결과를 학습에 반영하므로, 행동 정책이 목표 정책과 동일하다는 점에서 On-Policy 학습 방식으로 구분됩니다.
