# Stanford Reinforcement Learning - CS234 Summary

## Lecture 2 - How to act given know how the world works. 

### Questions for Anki

1. In what type of environment you need a discount rate < 1?
    * When the number of episidoes is infinite.
2. Why do you need a discount rate < 1 when the number of espisodes
is infinite?
    * To limit my value function. It can't be infinite too.
3. What is Markov Process?
    * Stochastic Process that satisfies the Markov Property.
    It has a state space and a transition probability model
4. What is a Markov Reward Process?
    * A Markov Process together with the specification of a reward
    function a discount factor
5. What is a Markov Decision Process?
    * A Markov Reward process with a set of actions an agent can take from 
    each state
6. What is the difference between the transition probability of a Markov Reward
process and a Markov Decision Process?
    * In the reward case, the transition probability map the probability of a 
    new state based on the current state. In the decision case, the transition 
    probability maps the probability of a new state based on the current state
    and the current action.
7. Define state-value function in the context of a markov decision process?
    * expected return starting from state s at time *t* following policy pi
8. What can you say about an MDP + a policy pi?
    * We have a markov reward process
9. How many policies we can have on an environment that has 7 discrete states
   and 2 discrete actions?
    * 2^7
