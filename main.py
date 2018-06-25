import gym
import tensorflow as tf
from Agent import Agent



if __name__ == "__main__":
    
    #PROBLEM = "CartPole-v0"
    PROBLEM = "LunarLander-v2"
    env = gym.make(PROBLEM)
    
    lr = 0.0001
    discount = 0.99
    eps = 0.1
    epsDiscount = 0.99
    
    
    updateSpeed = 1e4
    memSize = 1e5
    batchSize = 64
    
    with tf.Session() as sess:
        agent = Agent(env,sess,updateSpeed = updateSpeed,lr = lr, discount = discount, eps = eps, epsDiscount = epsDiscount, batchSize = batchSize, memSize = memSize )
        
        init = tf.global_variables_initializer()
        sess.run(init)
        agent.solveProblem()
        
        
        
        
        
        
        