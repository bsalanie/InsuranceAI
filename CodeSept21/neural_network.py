import tensorflow as tf

xi = tf.keras.initializers.he_normal()
zi = tf.keras.initializers.Constant(value=0.01)
ui = tf.keras.initializers.RandomUniform(minval=-0.001, maxval=0.001)

def create_neural_network(model_parameters):
    
    layer_tensors = []
    previous_size = model_parameters["input_size"]
    variables = []
    
    # Create layers
    for size, activation_function in model_parameters["layers"]:
        w = tf.Variable(ui((previous_size, size)))#, dtype=tf.float32)
        b = tf.Variable(zi((1, size)))#, dtype=tf.float32)
        
        layer_tensors.append((w, b, activation_function))
        variables.append(w)
        variables.append(b)
        previous_size = size
    
    @tf.function
    def apply_network(x, training):
        
        forward_propagation = x
        last_before_activation = None
        
        for i, (w, b, activation_function) in enumerate(layer_tensors):
            # Dropout disable
            if model_parameters["dropout"] is not None and training:
                w = tf.nn.dropout(w, model_parameters["dropout"])
                b = tf.nn.dropout(b, model_parameters["dropout"])
            
            forward_propagation = tf.add(tf.matmul(forward_propagation, w), b)
            last_before_activation = forward_propagation
            if activation_function is not None:
                forward_propagation = activation_function(forward_propagation)
        return last_before_activation, forward_propagation
        
    return apply_network, variables, layer_tensors