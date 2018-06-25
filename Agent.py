import numpy as np 
from Model import Model
from copy import copy
from Memory import Memory
import Utils






class Agent():
    
    
    def __init__(self,env,sess,updateSpeed = 5e4,lr = 0.0001,eps = 0.2,discount = 0.99,epsDiscount = 0.99, memSize = 1e5,batchSize = 64):
        self.env = env
        self.aSize = env.action_space.n
        self.sSize = env.observation_space.shape[0]
        
        self.lr = lr
        self.eps = eps
        self.discount = discount
        self.epsDiscount = epsDiscount
        self.updateSpeed = updateSpeed
        
        self.model = Model(self.aSize,self.sSize,"learning",lr = lr)
        self.target = Model(self.aSize,self.sSize,"target",lr = lr)
        self.sess = sess
        self.memory = Memory(memSize)
        self.batchSize = batchSize
        
        self.fillMemory()
    
    
    
    def fillMemory(self):
        while not self.memory.isFull():
            s = copy(self.env.reset())
            done = False
            while not done:
                a = np.random.randint(self.aSize)
                s_,r,done,_ = self.env.step(a)
                
                transition = np.array([s,a,r,s_,done])
                
                self.memory.addTransition(transition)
                
                s = s_
    
    
    def train(self):
        samples = self.memory.getNSamples(self.batchSize)
        s,a,r,s_,d = samples[0],samples[1],samples[2],samples[3],samples[4]               ##here change to array
        
        r = np.array(r,dtype = np.float32)
        a = np.array(a,dtype = np.int32)
        
        s = np.vstack(s)
        s_ = np.vstack(s_)
        r = np.vstack(r)
        
        
        
        
        notDoneInd = np.where(d == False)[0]
        doneInd = np.where(d == True)[0]
        Qs_ = np.zeros((self.batchSize,1))
        
        
        if notDoneInd.size != 0:
            feed_dict = {self.target.inputs:s_[notDoneInd]}
            Qs_tmp = self.sess.run(self.target.outNonSqueezed,feed_dict = feed_dict) 

            tmp = np.vstack(np.amax(Qs_tmp,axis = 1))
            
            Qs_[notDoneInd] = tmp
        
        elif doneInd.size != 0:
            Qs_[doneInd] = 0
        
        target = (r + self.discount * Qs_)
        

        self.model.train(s, target, a, self.sess)   
    
    def pickActionEps(self,Qs):
        if np.random.rand() < self.eps:
            return np.random.randint(self.aSize)
        else:
            return self.pickAction(Qs)
    
    
    
    def pickAction(self,Qs):
        maxim = np.amax(Qs)
        ind = np.where(Qs == maxim)[0]
        return np.random.choice(ind)
    
    
    
    
    
    def solveProblem(self):
        epCounter = 0
        stepCounter = 0
        while True:
            R = 0
            s = copy(self.env.reset())
            done = False
            while not done:
                stepCounter += 1
                
                if stepCounter == self.updateSpeed:
                    stepCounter = 0
                    ops = Utils.copyTrainableGraph("target", "learning")
                    self.sess.run(ops)
                    
                    
                if epCounter%15==0:  
                    self.env.render()
           
                feed_dict = {self.model.inputs:[s]}
                Qs = self.sess.run(self.model.out,feed_dict = feed_dict)

                
                    
                a = self.pickActionEps(Qs)
                
                s_,r,done,_ = self.env.step(a)
                
                s_ = copy(s_)
                R += r
                if R == 200:
                    transition = np.array([s,a,r,s_,False])
                else:
                    transition = np.array([s,a,r,s_,done])
                
                
                
                transition = np.array([s,a,r,s_,done])
                
                self.memory.addTransition(transition)
                
                self.train()                     
                
                s = s_
            
            if epCounter%1==0:    
                print("##################")
                print("last Qs: ",Qs)
                #print("stable Position Qs: ",QsStable)
                print("reward for ", epCounter, "-th episode is : ",R)
                print("##################")
            epCounter += 1
            
                
                
                 
                 
                
                 
        
        
        


