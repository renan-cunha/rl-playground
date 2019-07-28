# Stanford Reinforcement Learning - CS234 Summary

## Table of Contents

1. [Lecture 1 - Introduction](#lecture1-introduction)

## Lecture 1 - Introduction

### Questions for Anki
1. What are the four main subelements of a RL system?
    * Policy, Reward Signal, Value Function and Model
2. What is a *policy*?
    * Mapping from states to actions
3. What is a *value function*?
    * Total amount of reward an agent can expect to accumulate over the future,
    starting from a specific state
4. What is a *model* of the environment?
    * Given an state-action pair, the model might predict the next state and
    reward
5. Define Markov Property
    * Say that the conditional probability distribution of future states of an
    environment depends only upon the present state.
6. What is the difference between RL and AI planning?
    * In AI planning is given a model of how the world works. In RL we are
      learning the model by try and error.

### Sutton Questions
1. Suppose, instead of playing against a random opponent, the
reinforcement learning algorithm described above played against itself, with both sides
learning. What do you think would happen in this case? Would it learn a different policy
for selecting moves?
    * **My answer**: Would try to draw
    * **Correct Answer**: Would alternate between good and bad move to always
    win
2. Many tic-tac-toe positions appear different but are really
the same because of symmetries. How might we amend the learning process described
above to take advantage of this? In what ways would this change improve the learning
process? Now think again. Suppose the opponent did not take advantage of symmetries.
In that case, should we? Is it true, then, that symmetrically equivalent positions should
necessarily have the same value? 
    * **My answer**: We could design the state space in a more simplified way. If the opponent
    did not take advantage of symmetries, we should not too, because him can
    have a difference policy for different states even if they are symetric.
    So, it is not true that we symmetrically equivalent positions should have
    the same value
3. Suppose the reinforcement learning player was greedy, that is,
it always played the move that brought it to the position that it rated the best. Might it
learn to play better, or worse, than a nongreedy player? What problems might occur?
    * **My answer**: We can learn better than him, since a greedy strategy can yield a
      suboptimal solution.
