# DQN
This is implementation of DQN, DDQN, dueling (D)DQN and importance sampling from replay memory. At the beginning there is basic ideas explanation with link to original papers. After that there are some examples of trained agents and how to run it. 

## Motivation and base idea (DQN - deep Q net)
### Bias
One of the obstacles to overcome in reinforcement learning is dependency of learning batches. This dependency comes from the fact, that agents learn from environment on the run. Batches are therefore consecutive states of environment and this leads to huge bias in learning (imagine learning only from past 10 minutes of life while not remembering anything else). 

### How to get better data
One way to achieve more IID kind of data is to use combination of Q-learning (1 step TD learning for action values) estimated by NN and replay memory. This approach is called deep Q nets (DQN). Replay memory contains experiences from past. Instead of using last N states as single batch, we sample transitions from memory. These experiences are much more independent on each other since we get experiences from different time steps (same as in real life, what u experienced hundred days ago isn't usually connected to what you experienced today). Experiments showed that this is indeed good strategy to reduce bias. Good to note is that this whole idea is inspired by human learning through dreaming. ![DQN paper](https://arxiv.org/pdf/1312.5602.pdf)


## DDQN(dueling DQN)
### Base state value
In plenty of states we don't care about action that much, there are all approximately same value. We can take basically any action and it doesn't have that much effect on results. In other states we might have multiple actions with same value but one is outstanding (positively/negatively). Neither of these would be any problem if the baseline value would be 0. In this case we have to learn that first action has value N, then second action has value N, then third had value N and so on. We can say that state has some base value (state value V) and action might not change it that much (whatever we do, we achieve similiar rewards). Aside from this, learning Q values directly leads to their overestimation. (more info in paper below)

### Speeding up and increasing stability
We can speed up this process by separating Q-value(action value) into 2 terms. Value of state V (being rich is yaaay) and advantage Adv - how much we improve or worsen on action (bein rich is yaay but combining it with action "burning all of your money" leads to poor results - heheh POOR much funny :D). To get Q values we add up state value and advantages for all possible actions. In practical example, we separate last layer of Q network into 2 parts -> first part returns only 1 value - state value, the other returns all advantages. We then simply back propagate. 

### Indistinguishability
Then we find out that it doesn't work well, but why? Backpropagation that is usually used to optimaze NN doesn't know that the 1 output is state value and the N outputs are advantages and can achieve results in many random ways (set the state V value to any pseudo random values and then compensate in Qs which doesn't alleviate any work from regular DQN). To overcome this problem we must enforce some connection between Advantages and State value (to create only one possible solution). This can be done by calculatiing Qs as V + Adv - max(Adv) -> this enforce ma Adv to be equal V and e creates connection between V and Advantages. Ohter similar approach is Qs = V + Adv - mean(Adv). These solutions actually speed up learning process. Note that this is quite similiar to lagrangian multipliers (maybe it could even be considered one, I never really used them..). ![DDQN paper](https://arxiv.org/pdf/1511.06581.pdf)

## Double (D)DQN
### High variance
Next problem we might seek to solve is high variance. Source of this problem is in ever changing Q-values (according to which we also act). We act on environment for several steps and then we do NN batch update. Every time we do this Q-values change -> we act differently -> Q-values change more -> we act more differently(actions have high variance) -> we achieve mixed (high variance) results and have hard time using NN to aproximate them. 

### Stabilazing actions
Solution to this is double (D)DQN. Instead of single NN we use two. One (that is used for acting on environment) is freezed for some period of time to achieve consistent actions. In the meantime we update the other to predict Q-values as good as possible. After some learning time the frozen ones parameters are moved closer to params of the other (we start to act more according to actual Q-values, but not fully). This is same idea as if we were at shooting range and we had to hit moving target. That would be hard, so instead we stop it for moment shoot, shoot, shoot at still target, then move a bit and then shoot again at still standing target -> increase in accuracy of shooting == increase in correcet Q-values estimate. ![double DQN paper](https://arxiv.org/pdf/1509.06461.pdf)

## Importance sampling
### Memory efficiency
Lastly we might like to increase efficiency of sampling. Some experiences are more valuable than others. When someone shows you how to drive this experience is much more valuable (from learning new stuff perspective) than memory of you drinking water. First one has much more information in it. This is why we might like to use importance sampling. 

### Priority estimation
First we need to estimate how much important something is. Transition used in Q-nets is state, action, reward, new_state. Valuable experiences are those in which estimate of reward + action value in first state is far from estimate of best Q-value in new_state. This means that during one step something big happened, something we didn't expect and we should learn from it, so next time Q value of action in first state is estimated correctly. Example of this is child that tries to touch fire. In first state before touching fire, he thinks that touching it will be very pleasant (after all its warm and cozy and soo coloury :3). THen he touches it and finds out that the new state including burnt hands is highly unpleasant (wise men might say: his day will be ruined and his dissapointent will be immesurable). This kind of experience is then processed in his brain much more densly to avoid such case.

### Sampling
After estimation of priorities we employ sampling strategy. In my case i used rank based sampling (details and formulas in paper below). First of all we need to create probability distribution function for all samples in our memory. This distribution should give samples with most valuable information bigger probability than to others but should be non-zero for every possible transition (after all, we woudn't have to store something that isn't even used). In Rank based case we use rank of transition compared to others based on priority. We then use param alpha as power to which this number raise and we gain kind of exponential distribution. We ofcourse have to normalize and stuff :D. 

### reducing bias
This hoevewer introduces bias, paper below therefore proposes way to compensate this with weights. Weights tell us how much given samples error should contribute to NN update (if we are more likely to sample X then it should contribute less). But to some degree we want this bias, that is what makes it "priority sampled" for gods sake! So another parameter  is introduced. Beta (value from 0 to 1) regulates these weights. At the start of the learning Beta is lower because we want to embrace experiences that surprised us (like little children, we should be amazed by everything and learn from it). At end of the learning beta is high, because we don't want to overfit on some random super high variance experience that happens once in lifetime(that would get sampled over and over again because its sooo different) and fail in all regular cases. (When we are old, we can't just randomly start to question whether 1+1 (in base 10 ;)) is indeed 2, that would mess up our next progress/results too much) ![prioritized experience replay paper](https://arxiv.org/pdf/1511.05952.pdf)

## Examples
Here are few videos from agents at the end of the learning process. I used to have plenty of graphs and stuff that could be compared to Papers, but I lost on some hard drive. In future I would love to find them and add them/create new ones and add them.

![breakout](https://github.com/DanielKarasek/DQN/blob/master/videos_readme/breakout.gif)
![pong](https://github.com/DanielKarasek/DQN/blob/master/videos_readme/pong.gif)


## How to run it and requirements
This whole project was done in python 3.6, all required libraries are inside requirements.txt. After that some further actions might be needed for different environments (Mujoco, ROM atari etc. how to! are explained in error message when u try to run environment of this kind). To get all possible arguments call main.py with --help param. Quick rundown of most important ones.:
* --env_id - environment id, u can find all possible environments on opent AI gym official doc site (given this apps runs on gym 0.12, some newer might not be viable).
* --play - instead of learning agent only plays -> u can make gifs/watch how it progressed during previous learning
* --beta - starting value of importance sample beta
* --len_beta - count of steps along which beta is moved towards 1
* --alpha - between usually between 0 and 1 - zero leads to uniform distribution, higher number to more exponential distribution
* --reload - whether to load old agents progress (all other params except for logdir are ignored) 
* --logdir - folder with saved progress in case of reload option (not just checkpoint but even agent and stuff)

Other params are pretty self explanatory. All of them are grouped into learn (learning params such as lr, network architecture-might create extra group in future, eps politics), memory(memory settings) and reload.

During the main loop you can initiate rendering with pressing S and then enter.
You can also change verbosity level(not sure how many there are maybe even just 1) by pressing V enter and then number.
And lastly u can stop learning at any time by pressing Q and enter.
