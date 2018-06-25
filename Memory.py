import numpy as np



class Memory():
    
    
    def __init__(self,size):
        self.size = size
        self.memoryArr = []
        
        
        
    def addTransition(self,transition):
        if self.isFull():
            self.memoryArr = self.memoryArr[1:]
            self.memoryArr.append(transition)
        else:
            self.memoryArr.append(transition)
    
    def isFull(self):
        if len(self.memoryArr) >= self.size:
            return True
        else: return False
    
    
    def getNSamples(self,N):
        
        lenArr = len(self.memoryArr)
        N = min(N,lenArr)
        if lenArr > 0:
            indices = np.random.choice(lenArr,N)
            return np.column_stack([self.memoryArr[ind] for ind in indices])
        else:
            return -1
        
        