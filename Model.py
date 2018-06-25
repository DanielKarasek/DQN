import tensorflow as tf
import tensorflow.contrib.slim as slim
from astor.rtrip import out_prep



class Model():
    
    
    def __init__(self,actionSize,stateSize,name,lr = 0.001,deltaClip = 1):
        self.actionSize = actionSize
        self.stateSize = stateSize
        
        self.optimizer = self.createOptimizer(lr)
        
        self.graph = self.buildGraph(name,deltaClip)
        
    
    def createOptimizer(self,lr):
        
        RMS = tf.train.RMSPropOptimizer(learning_rate = lr)
        return RMS

    def buildGraph(self,name,deltaClip):
        with tf.variable_scope(name):
            self.inputs = tf.placeholder(tf.float32,shape = (None,self.stateSize),name = "inputs")                                ##input train/predict
            
            hidden = slim.fully_connected(inputs = self.inputs,num_outputs = 64 ,activation_fn = None)
            
            hidden2 = slim.fully_connected(inputs = hidden,num_outputs = 64 ,activation_fn = tf.nn.relu)
            
            out = slim.fully_connected(inputs = hidden2,num_outputs = self.actionSize,activation_fn = None)        ##output predict Q
            
            self.outNonSqueezed = out
            
            self.out = tf.squeeze(out)
            
            if name != "target":
            
                self.targets = tf.placeholder(tf.float32,shape = ((None,1)),name = "targets")                             ##input train
                self.actions = tf.placeholder(tf.int32,shape = (None),name = "actions")                                             ##input train
                                                                                                                
                
                actionOneHot = tf.one_hot(self.actions,self.actionSize,axis = -1)
                
                tdError = tf.abs(self.targets-out)
                ## self.tdErrorClipped = tf.clip_by_value(tdError, -1, 1, name = "ErrorClip")
                
                if deltaClip > 0:
                    quadraticPart = tf.clip_by_value(tdError, 0, deltaClip)
                    linearPart = tdError - quadraticPart
                    errors = (0.5 * tf.square(quadraticPart) + deltaClip * linearPart)
                    self.responsibleOuts = errors* actionOneHot
                    self.loss = tf.reduce_mean(self.responsibleOuts,axis = 0)  
                    
                else:
                    self.squaredDiff = tf.square(tdError)
                    self.responsibleOuts = self.squaredDiff* actionOneHot
                    self.loss = tf.reduce_mean(self.responsibleOuts,axis = 0)  
                
                self.trainStep = self.optimizer.minimize(self.loss,name = "trainStep")
        
        
    def train(self,inputs,target,actions,sess):
        inputs = inputs
        target = target
        actions = actions
        feed_dict = {self.inputs:inputs,self.targets:target,self.actions : actions}
        sess.run(self.trainStep,feed_dict = feed_dict)
        pass
        
        