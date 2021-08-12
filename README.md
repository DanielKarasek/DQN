# DQN
This is implementation of DQN, DDQN and importance sampling from replay memory.

## What is it? (DQN)
One of the obstacles to overcome in reinforcement learning is dependency of learning batches. This dependency comes from the fact, that agents learn from environment on the run. Batches are therefore consecutive states of environment and this leads to huge bias in learning (imagine learning only from past 10 minutes of life and then forgetting everything else). 
One way to achieve more IID kind of data is to use combination of Q-learning (1 step TD learning for action values) estimated by NN and replay memory. This approach is called deep Q nets (DQN). Replay memory contains experiences from past. Instead of using last N states as single batch, we sample transition from much further past experiences. These experiences are much more independent on each other (same as in real life, what u experienced hundred days ago isn't usually connected to what you experienced today). Experiments showed that this is indeed good strategy to reduce bias. Good to note is that this whole idea is inspired by human learn through dreaming.

## DDQN
Next problem we might seek to solve is high variance. Source of this problem is in ever changing Q-values (according to which we also act). We act on environment for several steps and then we do NN batch update. Every time we do this Q-values change -> we act differently -> Q-values change more -> we act more randomly(with high variance) -> we achiave mixed (high variance) results. Solution to this is DDQN aka dueling DQN. Instead of single NN we use two. One (one used for acting on environment) is freezed for some period of time to achieve consistent actions. In the meantime we update the other to predict Q-values as good as possible. Then we update the frozen with params from the other. This is same idea as if we were at shooting range and we had to hit moving target. That would be hard, so instead we stop it for moment shoot, shoot, shoot at still target, then move a bit and then shoot again at still standing target -> increase in accuracy of shooting == increase in correcet Q-values estimate. 

## Importance sampling
Lastly we might like to increase efficiency of sampling. Some experiences are more valuable than others. When someone shows you how to drive this experience is much more valuable (from learning new stuff perspective) than memory of you drinking water. First one has much more information in it. This is why we might like to use importance sampling. First we need to estimate how much important. Transition used in Q-nets is state, action, reward, new_state. Valuable experiences are those in which estimate of reward + action value in first state is far from estimate of best Q-value in new_state. This means that during one step something big happened, something we didn't expect and we should learn from it, so next time Q value of action in first state is estimated correctly. Example of this is child that tries to touch fire. In first state before touching fire, he thinks that touching it will be very pleasant. THen he touches it and finds out that the new state including burnt hands is highly unpleasant. This kind of experience is then processed in his brain much more densly to avoid such case.
We then employ sampling strategy, in my case i used rank based sampling (details and formulas in paper below). First of all we need to create probability distribution function for all samples in our memory. This distribution should give samples with most valuable information bigger probability than to others but should be non-zero for every sample. In Rank based case we use rank of transition compared to others based on priority. We then use param alpha as power to which this number raise and we gain kind of exponential distribution. We ofcourse have to normalize and stuff :D.

This hoevewer introduces bias, paper below therefor proposes way to compensate this with weights. Weights tell us how much given samples error should contribute to NN update (if we are more likely to sample X then it should contribute less). But to some degree we want this bias, that is what makes it "priority sampled"! Another parameter Beta (value from 0 to 1) regulates these weights. At the start of the learning Beta is lower because we want to embrace experiences that surprised us. At end of the learning beta is high, because we don't want to overfit on some random super high variance experience that happens once in lifetime(that would get sampled over and over again because its sooo different) and fail in all regular cases.

## Examples
Here are few videos from agents at the end of the learning process. I used to have plenty of graphs and shi.. which could be compared to Papers. In future I would love to find them and add them/create new ones and add them.

https://user-images.githubusercontent.com/33940762/129268418-4ac59032-1e5e-4bfb-b815-5ddf31bf752c.mp4
https://user-images.githubusercontent.com/33940762/129268433-981849bb-27e9-4a32-a058-8514aed83f16.mp4



## Parameters and how to run it
