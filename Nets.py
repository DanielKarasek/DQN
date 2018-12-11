import tensorflow as tf




def build_dnn(net,action_size,layers = [128,32]):
    '''
    Function creating DNN with with elu activations 
    in hidden layers and linear activation for output
    layer.
    args:
        net: integer - inputs to neural network
        action_size: integer - number of outputs
        layers: Iterable object of scalar numbers where each number represents
               number of units in given layer e.g.2 hidden layers with 10 and 15 neurons == [10,15])
    
    returns: Handle to output layer
    '''

    with tf.variable_scope("DNN_Net"):
        for idx,units in enumerate(layers):
            net = tf.layers.dense(inputs=net,
                                  units=units,
                                  activation=tf.nn.elu,
                                  name="hidden_"+str(idx),
                                  )
            
        return tf.layers.dense(inputs = net,
                               units = action_size,
                               activation = None,
                               name = "out",
                               )

            

def build_cnn():
    
    pass