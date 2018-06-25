import tensorflow as tf


'''
    Takes in names of Models. 
    Then uses theirs trainable variables to create
    operations for copy which must be then run with tf Session
'''

def copyTrainableGraph(destModelName,srcModelName):
    srcVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,srcModelName)
    destVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,destModelName)
    
    ops = []
    
    for srcVar,destVar in zip(srcVars,destVars):
        ops.append(tf.assign(destVar, srcVar))
    return ops
    