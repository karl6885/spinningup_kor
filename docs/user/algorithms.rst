==========
Algorithms
==========

.. contents:: 목차

무엇이 포함되어 있나
===============

Spinning Up 패키지에는 다음과 같은 알고리즘들이 구현되어 있다:

- `Vanilla Policy Gradient`_ (VPG)
- `Trust Region Policy Optimization`_ (TRPO)
- `Proximal Policy Optimization`_ (PPO)
- `Deep Deterministic Policy Gradient`_ (DDPG)
- `Twin Delayed DDPG`_ (TD3)
- `Soft Actor-Critic`_ (SAC)

이 알고리즘들은 모두 `MLP`_ (non-recurrent) actor-critics로 구현되어 있으며, `Gym Mujoco`_ 환경과 같이 fully-observed, non-image-based인 RL 환경에 적합하도록 만들어졌다.

.. _`Gym Mujoco`: https://gym.openai.com/envs/#mujoco
.. _`Vanilla Policy Gradient`: ../algorithms/vpg.html
.. _`Trust Region Policy Optimization`: ../algorithms/trpo.html
.. _`Proximal Policy Optimization`: ../algorithms/ppo.html
.. _`Deep Deterministic Policy Gradient`: ../algorithms/ddpg.html
.. _`Twin Delayed DDPG`: ../algorithms/td3.html
.. _`Soft Actor-Critic`: ../algorithms/sac.html
.. _`MLP`: https://en.wikipedia.org/wiki/Multilayer_perceptron


왜 이 알고리즘들인가?
=====================

우리는 위와 같은 핵심 deep RL 알고리즘들을 선택해 최근 이 분야에서 나타난 아이디어들의 유용한 발전들을 반영하고자 했다. 특히 PPO와 SAC, 이 두 알고리즘은 policy-learning 알고리즘들 중에서 reliability 측면이나 sample efficiency 측면에서 SOTA에 근접한 알고리즘들이다. 또한 이 알고리즘들은 deep RL 알고리즘을 만들고 사용함에 있어 발생할 수 있는 상충관계(trade-off)들을 보여주기도 한다.

The On-Policy Algorithms
------------------------

Vanilla Policy Gradient은 deep RL에 있어 가장 기초가 되는 알고리즘인데, 바로 deep RL의 완전한 부모 격인 알고리즘이기 때문이다. VPG의 핵심 요소들이 등장한 시기는 80년대 후반/90년대 초반까지 거슬러 올라간다. 이것은 이후 TRPO와 PPO와 같은 강력한 알고리즘으로 이어지는 연구 흐름의 시작이 된다.

이러한 연구 흐름의 특징은 바로 해당 알고리즘들이 모두 *on-policy* (정책 기반) 알고리즘이라는 것이다. 즉, 이 알고리즘들은 과거 데이터를 쓰지 않으며, 이로 인해 sample efficiency 측면에서 약한 모습을 보인다. 그러나 여기에는 그럴만한 이유가 있는데, 이 알고리즘들은 사용자가 달성하고자 하는 목표(policy performance, 정책의 성능)를 직접적으로 최적화하며, 이것이 수학적으로 작동하는데 업데이트를 계산하기 위해서는 on-policy 데이터를 필요로 하기 때문이다. 따라서 이 계열의 알고리즘들은 stability를 얻는 대신 sample efficiency를 잃는 상충관계를 가지고 있다. 그렇지만 그렇게 잃은 sample efficiency를 보충하고자 VPG부터 TRPO, PPO로 기술들의 발전이 이어졌다는 것을 확인할 수 있다.


The Off-Policy Algorithms
-------------------------

DDPG는 VPG와 비슷하게 기반이 되는 알고리즘이긴 하지만, 훨씬 최근에 등장한 편이다. 추후 DDPG로 이어지는 deterministic policy gradients 이론은 2014년에야 출판되었다. DDPG는 Q-learning 알고리즘들과 밀접하게 연결되어 있는데, Q-function과 policy가 서로를 개선시키도록 업데이트하는 방식으로 함께 학습한다.

DDPG와 Q-Learning과 같은 알고리즘들은 *off-policy*로, 과거의 데이터들을 굉장히 효율적으로 재활용한다. 이 알고리즘들은 Bellman's equations for optimality를 활용하여 이러한 장점을 얻을 수 있는데, 해당 environment의 high-reward area에서 충분한 experience가 있다면 Q-function이 *어떠한* environment interaction data를 활용해서라도 학습할 수 있기 때문이다.

그렇지만 문제가 되는 부분은, Bellman's equations를 만족시켜 학습시키는 것이 훌륭한 policy performance라는 결과로 이어질지 장담할 수 없다는 것이다. *경험적으로* 누군가는 훌륭한 결과를 얻을 수 있겠지만, 그리고 일단 그렇기만 하면 sample efficiency가 좋겠지만, 그러한 결과를 장담할 수 없기 때문에 이 계열의 알고리즘들을 잠재적으로 불안정하다. TD3와 SAC는 DDPG의 후속 알고리즘들로 이러한 문제를 해결하기 위해 다양한 방법을 활용한다.

Code Format
===========

Spinning Up에 있는 모든 구현체들은 표준 템플릿을 따른다. 각 구현체들은 2개의 file로 나뉜다: 알고리즘의 코어 로직을 담고 있는 algorith file과, 알고리즘을 실행하기 위한 다양한 유틸리티들을 담고 있는 core file이다.

The Algorithm File
------------------

algorithm file은 experience buffer 객체를 위한 class 정의로 시작한다. 이것은 agent-environment interaction에서 나온 정보를 저장하는 데 쓰인다.

다음은, 알고리즘을 실행하기 위한 하나의 function이 있으며, 다음과 같은 작업을 순서대로 한다:

    1) Logger setup

    2) Random seed setting
    
    3) Environment instantiation
    
    4) Making placeholders for the computation graph
    
    5) Building the actor-critic computation graph via the ``actor_critic`` function passed to the algorithm function as an argument
    
    6) Instantiating the experience buffer
    
    7) Building the computation graph for loss functions and diagnostics specific to the algorithm
    
    8) Making training ops
    
    9) Making the TF Session and initializing parameters
    
    10) Setting up model saving through the logger
    
    11) Defining functions needed for running the main loop of the algorithm (eg the core update function, get action function, and test agent function, depending on the algorithm)
    
    12) Running the main loop of the algorithm:
    
        a) Run the agent in the environment
    
        b) Periodically update the parameters of the agent according to the main equations of the algorithm
    
        c) Log key performance metrics and save agent


마지막으로, 커맨드 라인으로 Gym environment에서 알고리즘을 직접 실행하도록 지원하는 부분이 있다.

The Core File
-------------

core file은 algorithm 파일만큼 엄격하게 템플릿을 준수하지는 않지만, 대략적인 구조는 있다:

    1) Functions related to making and managing placeholders

    2) Functions for building sections of computation graph relevant to the ``actor_critic`` method for a particular algorithm

    3) Any other useful functions

    4) Implementations for an MLP actor-critic compatible with the algorithm, where both the policy and the value function(s) are represented by simple MLPs


