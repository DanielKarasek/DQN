# DQN
This is implementation of DQN, DDQN and importance sampling from replay memory.

## What is it? (DQN)
One of the obstacles to overcome in reinforcement learning is dependency of learning batches. This dependency comes from the fact, that agents learn from environment on the run. Batches are therefore consecutive states of environment and this leads to huge bias in learning. 
One way to achieve more IID kind of data is to use combination of Q-learning (1 step TD learning for action values) estimated by NN with replay memory (this is called deep Q nets aka DQN). Replay memory contains experiences from past. Instead of using last N states as single batch, we can sample transitions used for NN error estimation from past experiences. These experiences are much more independent on each other (same as in real life, what u experienced hundred days ago isn't usually connected to what you experienced today). Experiments showed that this is indeed good strategy to reduce bias. Good to note is that this whole idea is inspired by human dreaming.

## DDQN
Next problem we might seek to solve is high variance. Source of this problem is in ever changing Q-values. We act on environment for several steps and then we do NN batch update. Every time we do this Q-values change -> we act differently -> Q-values change more, because we are off more. Solution to this is DDQN aka dueling DQN. Instead of single NN we use two. One (one used for acting on environment) is freezed for some period of time to achieve consistent actions. In the meantime we update the other to predict Q-values as good as possible. This is same idea as if we were at shooting range and we had to hit moving target. That would be hard, so instead we stop it for moment shoot, shoot, shoot, then move a bit and then shoot again at still standing target -> increase in accuracy of shooting == increase in Q-values estimate. 

## Importance sampling
Lastly we might like to increase efficiency of sampling. Some experiences are more valuable than others. When someone shows you how to drive this experience is much more valuable (from learning new stuff perspective) than memory of you drinking water. First one has much more information in it. This is why we might like to use importance sampling. First we need to estimate how much important. Transition used in Q-nets is state, action, reward, new_state. Valuable experiences are those in which estimate of reward + action value in first state is far from estimate of best Q-value in new_state. This means that during one step something big happened, something we didn't expect and we should learn from it, so next time Q value of action in first state is estimated correctly. Example of this is child that tries to touch fire. In first state before touching fire, he thinks that touching it will be very pleasant. THen he touches it and finds out that the new state including burnt hands is highly unpleasant. This kind of experience is then processed in his brain much more densly to avoid such case.
We then need to employ sampling strategy.




## 

## Parameters and how to run it

## Future additions
