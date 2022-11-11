#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:

from paths import *
from utilities import *

def run(experiment_name, rho_test, power_test_or_bootstrap_choice, mu_4_choice, skip_estimating_fhat, enable_one_hot):


    # Python library
    import math
    import statistics
    from abc import abstractmethod
    import pickle

    # Machine learning
    import tensorflow as tf
    import numpy as np
    import matplotlib.pyplot as plt
    from tensorflow.keras import layers
    from sklearn.linear_model import LinearRegression

    init_plt()
    init_tf()
    tf.keras.backend.set_floatx('float32')

    # # Parameters

    # In[3]:
    NUMBER_OF_FEATURES = 50 #9

    # All parameters dict
    p = {
        "experiment_type": "constant_rho",
        "experiment_name": experiment_name,
        "skip_estimating_fhat": skip_estimating_fhat,
        "enable_one_hot": enable_one_hot,
        "s": 102, # power_test s should be 1 more than the usual because the first is thrown out
        "k": 5,
        "rho_test": rho_test,
        "power_test_or_bootstrap": ["power_test", "bootstrap"][power_test_or_bootstrap_choice], # Choose either 0 or 1
        "negative_samples": ["delete", "set_to_zero"][0], # Choose either 0 or 1
        "mu_4_choice": ["simple", "hard", "none"][mu_4_choice], # Choose 0-2
        "select_young": False,
        "select_old": False,
        "override_weights_to_one": False,
        "eps": 0.0,

        "models_layers_and_parameters": {
            "weights": {
                "input_size": NUMBER_OF_FEATURES, # Do not change
                "layers":[
                    # Format: (Size, Activation Function (None if none))
                    # Last (output) layer should have size 1
                    #(128, tf.nn.sigmoid),
                    #(64, tf.nn.sigmoid),
                    (32, tf.nn.sigmoid),
                    (32, tf.nn.sigmoid),
                    (1, None)
                ],
                "learning_rate": 1e-3,
                "training_steps": 5000,
                "dropout": 0.2,
                "display_step": 1000,

            },

            "fhat": {
                "input_size": NUMBER_OF_FEATURES, # Do not change
                "layers":[
                    # Format: (Size, Activation Function (None if none))
                    # Last (output) layer should have size 1
                    #(128, tf.nn.sigmoid),
                    #(64, tf.nn.sigmoid),
                    (32, tf.nn.sigmoid),
                    (32, tf.nn.sigmoid),
                    (4, None)
                ],
                "learning_rate": 1e-3,
                "training_steps": 5000,
                "dropout": 0.2,
                "display_step": 1000,

                "resample_data": False,
                "resampled_points_per_category": 10000,
            }
        }
    }


    assert(not (p["select_young"] and p["select_old"])) # Make sure both are not selected
    OUTPUT_FOLDER = GENERAL_RESULTS_FOLDER + p["experiment_name"] + SLASH
    DISPLAY_STEP = 10000


    # # Make the folder where the results will be placed in

    # In[4]:


    get_ipython().system(' mkdir $OUTPUT_FOLDER')


    # # Data preparation

    # In[5]:


    # Load entire data file
    data = np.loadtxt(DATA_PATH).astype(np.float32)

    # Shuffle data
    np.random.shuffle(data)

    young_mask = data[:, 0]==1
    old_mask = data[:, 0]==2

    # Select young or old
    if p["select_young"]:
        #print("We selected only young.")
        data = data[young_mask]
    if p["select_old"]:
        #print("We selected only old.")
        data = data[old_mask]


    # In[6]:


    # Divide data into training, test, cross_validation
    samples_per_fold = math.floor(data.shape[0]/p["k"])
    samples = samples_per_fold*p["k"]
    #print("Samples:", samples)
    bounds = list(map(lambda x: (x, x+samples_per_fold), list(range(0, samples_per_fold*p["k"], samples_per_fold))))

    data = data[:p["k"]*samples_per_fold] # This is all we can use
    young_mask = young_mask[:p["k"]*samples_per_fold]
    old_mask = old_mask[:p["k"]*samples_per_fold]


    # In[7]:


    # Extract features, y_1, y_2, weights

    number_of_samples = data.shape[0]

    if p["enable_one_hot"]:
        def make_one_hot(x):
            unique = list(set(x))
            convert_to_index = np.vectorize(lambda x: unique.index(x))

            one_hot = np.zeros((number_of_samples, len(unique)))
            one_hot[np.arange(number_of_samples), convert_to_index(x)] = 1

            one_hot = one_hot[:, :-1]
            return one_hot
    else:
        def make_one_hot(x):
            return np.expand_dims(x, axis=1) # Do nothing, just add a dimension


    def extract(data):

        age = (data[:, 0]==2).astype(np.float32)
        group = data[:, 21] # 
        hom = data[:,23]
        prof  = data[:, 43]; #
        reg = data[:, 46] #
        rena = data[:, 47]
        tracir = data[:, 52] #
        trage = data[:, 53] # 
        usag = data[:,56] # 
        zone = data[:, 57] #

        y_1 = (data[:, 9] > 0).astype(np.float32)
        y_2 = (data[:, 24] == 0).astype(np.float32)
        weights = data[:, 42]

        # Convert These variables to ones and zeros
        list_categorical_variables =         list(map(make_one_hot, [group, prof, reg, tracir, trage, usag, zone]))

        # Do not convert these (they only take values 0 and 1)
        add_dim = lambda x: np.expand_dims(x, axis=1)
        list_other_variables =         list(map(add_dim, [hom, rena, age]))


        # Just add here
        return np.concatenate(list_categorical_variables+list_other_variables, axis=1),            add_dim(y_1),            add_dim(y_2),            add_dim(weights)

    X, y_1, y_2, weights = extract(data)
    X = X.astype(np.float32)
    y_1 = y_1.astype(np.float32)
    y_2 = y_2.astype(np.float32)
    #print(X.shape)

    # In[8]:


    # Normalize X
    
    if p["enable_one_hot"]:
        pass
    else:
        divide_by_std_dev = lambda x: x/np.std(x, axis=0, keepdims=True)
        subtract_mean = lambda x: x-np.mean(x, axis=0, keepdims=True)

        X = divide_by_std_dev(subtract_mean(X))


    # In[9]:


    t = lambda x: np.squeeze(x, axis=1)

    f00 = ((y_1 == 0) & (y_2 == 0)).astype(np.float32)
    f01 = ((y_1 == 0) & (y_2 == 1)).astype(np.float32)
    f10 = ((y_1 == 1) & (y_2 == 0)).astype(np.float32)
    f11 = ((y_1 == 1) & (y_2 == 1)).astype(np.float32)

    f = np.empty((y_1.shape[0], 4), dtype=np.float32)
    f[:, 0] = t(f00)
    f[:, 1] = t(f01)
    f[:, 2] = t(f10)
    f[:, 3] = t(f11)


    # In[10]:

    from neural_network import create_neural_network
    class NNModel():

        def __init__(self, model_layers_and_parameters):
            self.parameters = model_layers_and_parameters
            apply_network, variables, _ = create_neural_network(model_layers_and_parameters)
            self.variables = variables
            self.nn_layers = apply_network
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=model_layers_and_parameters["learning_rate"])

        @tf.function(experimental_relax_shapes=True)
        @abstractmethod # Means this function must be implemented
        def run(self, inp, training):
            pass 

        def train_one_step(self, inp):
            with tf.GradientTape() as tape:
                output = self.run(inp, training=True)

            loss = output["loss"]
            grads = tape.gradient(loss, self.variables)
            self.optimizer.apply_gradients(zip(grads, self.variables)) 

            return output

        def train(self, inp):

            for i in range(self.parameters["training_steps"]):
                output = self.train_one_step(inp)

                #if i % DISPLAY_STEP == 0:
                    #print("Step", i, "loss is", output["loss"].numpy())

            return output


    # # Models

    # In[11]:


    class WeightModel(NNModel):

        @tf.function(experimental_relax_shapes=True)
        def run(self, inp, training):
            _, output_unsigmoided = self.nn_layers(inp["X"], training=training)
            output_sigmoided = tf.sigmoid(output_unsigmoided)
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=inp["weights"], logits=output_unsigmoided)
            loss = tf.reduce_mean(cross_entropy)
            return {
                "weights_predicted": output_sigmoided, 
                "loss": loss
            }

    class FhatModel(NNModel):

        @tf.function(experimental_relax_shapes=True)
        def run(self, inp, training):

            _, unsoftmaxed = self.nn_layers(inp["X"], training=training)
            softmaxed = tf.nn.softmax(unsoftmaxed, axis=1)
            weights_squeezed = tf.squeeze(inp["weights"], axis=1)


            entropied = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.math.argmax(inp["f"], axis=1),                                                                    logits=unsoftmaxed)
            loss = tf.reduce_mean(weights_squeezed*entropied)
            accuracy = 100.0*tf.reduce_mean(tf.cast(tf.math.equal(tf.math.argmax(softmaxed, axis=1),                                                               tf.math.argmax(inp["f"], axis=1)), tf.float32))

            softmaxed_again = softmaxed / tf.reduce_sum(softmaxed, axis=1, keepdims=True)
            f_list = tf.unstack(tf.expand_dims(softmaxed_again, axis=2), axis=1)
            fhat00 = f_list[0]
            fhat01 = f_list[1]
            fhat10 = f_list[2]
            fhat11 = f_list[3]

            return {
                "unsoftmaxed": unsoftmaxed,
                "softmaxed": softmaxed,
                "accuracy": accuracy,
                "loss": loss,
                "fhat00": fhat00, 
                "fhat01": fhat01, 
                "fhat10": fhat10, 
                "fhat11": fhat11,
            }


    # # Rho analysis function (needed later, cannot divide into pieces because it runs inside a loop). This is the code with formulas that could be incorrect.

    # In[12]:


    import numpy as np
    from sklearn.linear_model import LinearRegression

    def dim(x, i):
        return np.expand_dims(x, i)

    def eval_omega(ro, weighti, phati, qhati, Rhati, fhat00i, fhat01i, fhat10i, fhat11i):
        Omega = np.zeros((4, 4))
        fmat = np.array([fhat00i, fhat01i, fhat10i, fhat11i]).reshape((2, 2))
        for j in [0, 1]:
            for k in [0, 1]:
                m = weighti*((j-phati)*(k-qhati)-ro*Rhati)
                m01 = weighti*((1-j)*k-fhat01i)
                m10 = weighti*(j*(1-k)-fhat10i)
                m11 = weighti*(j*k-fhat11i)
                mjk = np.array([m01, m10, m11, m])
                Omega += fmat[j, k]*np.outer(mjk, mjk)
        return Omega

    def rho_analysis(X, y_1, y_2, f00, f01, f10, f11, fhat00, fhat01, fhat10, fhat11, weights, weights_original, mu_4_choice):

        phat = fhat10+fhat11
        qhat = fhat01+fhat11
        Rhat = np.sqrt(phat*(1-phat)*qhat*(1-qhat))
        Rhat01 = phat*(1-phat)*(1-2*qhat)/(2*Rhat)
        Rhat10 = qhat*(1-qhat)*(1-2*phat)/(2*Rhat)
        Rhat11 = Rhat01+Rhat10

        #m = weights_original*((y_1-phat)*(y_2-qhat)-ro*Rhat)
        m01 = weights_original*((1-y_1)*y_2-fhat01)
        m10 = weights_original*(y_1*(1-y_2)-fhat10)
        m11 = weights_original*(y_1*y_2-fhat11)

        ro_naive = (fhat11-qhat*phat)/Rhat
        lr = LinearRegression()
        _ = lr.fit(X, ro_naive)
        predicted_ro_naive = lr.predict(X)

        ro_0 = weights_original*(fhat11-qhat*phat)/(weights_original*Rhat+Rhat01*m01+Rhat10*m10+Rhat11*m11)    
        lr = LinearRegression()
        _ = lr.fit(X, ro_0)
        predicted_ro_0 = lr.predict(X)

        ro_1 = weights_original*(y_1-phat)*(y_2-qhat)/(weights_original*Rhat+Rhat01*m01+Rhat10*m10+Rhat11*m11)
        lr = LinearRegression()
        _ = lr.fit(X, ro_1)
        predicted_ro_1 = lr.predict(X)

        #t = lambda x: np.abs(x / (weights_original*Rhat))
        #term1 = t(Rhat01*m01) 
        #term2 = t(Rhat10*m10)
        #term3 = t(Rhat11*m11)

        samples = y_1.shape[0]
        #smooth_weights = weights
        #smooth_weights2 = weights*weights

        # Loop and I and J
        def execute_ro_loop(mu_function):
            ro_loop = -1
            ro_hat = 777
            while abs(ro_loop-ro_hat)>0.0001:
                mu_4 = mu_function(ro_loop)
                ro_hat = np.sum(mu_4*weights*(y_1-phat)*(y_2-qhat)) / np.sum(mu_4*(weights*Rhat+Rhat01*m01+Rhat10*m10+Rhat11*m11))
                ro_loop = ro_loop*0.50+ro_hat*0.50
            return ro_loop
        def compute_psi(ro, mu_4_function):
            m = weights*((y_1-phat)*(y_2-qhat)-ro*Rhat)
            psi = mu_4_function(ro)*(m-ro*(Rhat01*m01+Rhat10*m10+Rhat11*m11))
            return psi
        def I_and_J(ro, mu_4_function):

            left  = 1-10e-3
            right = 1+10e-3
            left_ro = ro*left
            right_ro = ro*right

            psi_left   = compute_psi(left_ro, mu_4_function)
            psi_center = compute_psi(ro, mu_4_function)
            psi_right  = compute_psi(right_ro, mu_4_function)

            I = np.mean(psi_center**2)
            J = np.mean((psi_right - psi_left)/(right_ro - left_ro))
            return I, J

        # Now we have different mu_4 algorithms

        def mu_function_simple(ro):
            return - weights*Rhat/(1+ro**2*(Rhat01**2+Rhat10**2+Rhat11**2))

        def mu_function_hard(ro):

            mu_4_vec = np.empty((samples, 1, 1), dtype=np.float32)
            for i in range(samples):
                fhat00i = fhat00[i]
                fhat01i = fhat01[i]

                fhat10i = fhat10[i]
                fhat11i = fhat11[i]
                phati = phat[i]
                qhati = qhat[i]
                phati = phat[i]
                Rhati = Rhat[i]
                Rhat01i = Rhat01[i]
                Rhat10i = Rhat10[i]
                Rhat11i = Rhat11[i]

                # here we should use E(w | X=x_i)
                weighti = weights[i]#smooth_weights[i]
                Omega = eval_omega(ro, weighti, phati, qhati, Rhati, fhat00i,
                                   fhat01i, fhat10i, fhat11i)

                OmegaInv = np.linalg.inv(Omega)


                A_prime = np.array([0.0, 0.0, 0.0, -float(Rhati*weighti)])
                KGE = -weighti*np.stack((np.array([1.0, 0.0, 0.0]),
                                         np.array([0.0, 1.0, 0.0]),
                                         np.array([0.0, 0.0, 1.0]),
                                         np.array([float(ro*Rhat01i), float(ro*Rhat10i), float(ro*Rhat11i)])), axis=0)
                G = dim((A_prime @ OmegaInv @ KGE),
                        1).T @ np.linalg.inv(KGE.T @ OmegaInv @ KGE)

                result = (A_prime - G @ KGE.T) @ Omega[:, 3]
                mu_4_vec[i] = float(result)


            return np.squeeze(mu_4_vec, axis=1)*weights


        def mu_function_none(ro):
            return weights*np.ones_like(weights)

        if "simple" == mu_4_choice:
            ro_loop = execute_ro_loop(mu_function_simple)
            I, J = I_and_J(ro_loop, mu_function_simple)
        elif "hard" == mu_4_choice:
            ro_loop = execute_ro_loop(mu_function_hard)
            I, J = I_and_J(ro_loop, mu_function_hard)
        elif "none" == mu_4_choice:
            ro_loop = execute_ro_loop(mu_function_none)
            I, J = I_and_J(ro_loop, mu_function_none)

        return {
            "rho_naive": ro_naive,
            "predicted_rho_naive": predicted_ro_naive,
            "rho_0": ro_0,
            "predicted_rho_0": predicted_ro_0,
            "rho_1": ro_1,
            "predicted_rho_1": predicted_ro_1,
            #"term1": term1,
            #"term2": term2, 
            #"term3": term3,
            "rho_loop": ro_loop,
            "I": I, 
            "J": J,
            "phat": phat,
            "qhat": qhat,
        }



    # # Weights model training

    # In[13]:


    # Weights model training
    # Remember the old ones
    if "weights_original" not in locals().keys(): # If we haven't defined weights_original already
        weights_original = weights 

    # Predict weights
    if p["override_weights_to_one"]:
        weights = np.ones_like(weights)
    else:
        weights = np.empty_like(weights)

        for k in range(p["k"]):

            lower, upper = bounds[k]
            get_kth = lambda x: x[lower:upper]
            cut_kth = lambda x: np.concatenate([x[:lower], x[upper:]], axis=0)
            X_kth, weights_kth = list(map(get_kth, [X, weights_original]))
            X_rest, weights_rest = list(map(cut_kth, [X, weights_original]))

            # Reset graph
            tf.keras.backend.clear_session()

            model = WeightModel(p["models_layers_and_parameters"]["weights"])

            inp = dict_to_tensors({"X": X_rest, "weights": weights_rest})
            _ = model.train(inp)

            inp = dict_to_tensors({"X": X_kth, "weights": weights_kth})
            output_kth = model.run(inp, training=False)
            #print("Final loss on kth fold:", output_kth["loss"].numpy())
            weights[lower:upper] = output_kth["weights_predicted"].numpy() # Replace weights with predicted ones


    # In[14]:


    # # Main Training

    # In[15]:


    rho_list = np.empty((p["s"],))
    sigma_list = np.empty((p["s"],))
    T_list = np.empty((p["s"],))
    I_list = np.empty((p["s"],))
    J_list = np.empty((p["s"],))

    for s in range(p["s"]):

        #print("S:", s)

        if p["skip_estimating_fhat"] and ((s >= 2 and p["power_test_or_bootstrap"]=="power_test") or
                                          (s >= 1 and p["power_test_or_bootstrap"]=="bootstrap")):
            indexes = np.random.randint(low = 0, high = samples, size = samples)
            X_s = X[indexes]
            weights_s = weights[indexes]
            weights_original_s = weights_original[indexes]
            y_1_s = y_1[indexes]
            y_2_s = y_2[indexes]
            f_s = f[indexes]

            # Use previous fhats
            fhat00 = fhat00_calculated[indexes]
            fhat01 = fhat01_calculated[indexes]
            fhat10 = fhat10_calculated[indexes]
            fhat11 = fhat11_calculated[indexes]
        else:

            if (s == 0 and p["power_test_or_bootstrap"]=="bootstrap") or            (s <= 1 and p["power_test_or_bootstrap"]=="power_test"):
                X_s, y_1_s, y_2_s, f_s, weights_s, weights_original_s = [X, y_1, y_2, f, weights, weights_original]
            elif (s > 0 and p["power_test_or_bootstrap"]=="bootstrap") or              (s > 1 and p["power_test_or_bootstrap"]=="power_test"):
                indexes = np.random.randint(low = 0, high = samples, size = samples)
                X_s = X[indexes]
                weights_s = weights[indexes]
                weights_original_s = weights_original[indexes]
                y_1_s = y_1[indexes]
                y_2_s = y_2[indexes]
                f_s = f[indexes]
            else:
                raise Exception("Should be one of the two options above.")


            f00_s = f_s[:, 0:1]
            f01_s = f_s[:, 1:2]
            f10_s = f_s[:, 2:3]
            f11_s = f_s[:, 3:4]

            fhats = {"fhat00":np.zeros((samples, 1), dtype=np.float32),
                     "fhat01":np.zeros((samples, 1), dtype=np.float32),
                     "fhat10":np.zeros((samples, 1), dtype=np.float32),
                     "fhat11":np.zeros((samples, 1), dtype=np.float32)}

            for k in range(p["k"]):

                # Reset graph
                tf.keras.backend.clear_session()

                #print("K:", k)

                # Cut out part of kth fold
                lower, upper = bounds[k]
                get_kth = lambda x: x[lower:upper]
                cut_kth = lambda x: np.concatenate([x[:lower], x[upper:samples]], axis=0)
                X_kth, f_kth, weights_kth = list(map(get_kth, [X_s, f_s, weights_original_s]))
                X_rest, f_rest, weights_rest = list(map(cut_kth, [X_s, f_s, weights_original_s]))

                if p["models_layers_and_parameters"]["fhat"]["resample_data"]:
                    resampled_points_per_category = p["models_layers_and_parameters"]["fhat"]["resampled_points_per_category"]

                    indexes00 = np.where(f_rest[:, 0] == 1)[0]
                    indexes01 = np.where(f_rest[:, 1] == 1)[0]
                    indexes10 = np.where(f_rest[:, 2] == 1)[0]
                    indexes11 = np.where(f_rest[:, 3] == 1)[0]
                    chosen00 = indexes00[np.random.randint(0, indexes00.shape[0], size=resampled_points_per_category)]
                    chosen01 = indexes01[np.random.randint(0, indexes01.shape[0], size=resampled_points_per_category)]
                    chosen10 = indexes10[np.random.randint(0, indexes10.shape[0], size=resampled_points_per_category)]
                    chosen11 = indexes11[np.random.randint(0, indexes11.shape[0], size=resampled_points_per_category)]

                    X_train = np.concatenate([
                        X_rest[chosen00],
                        X_rest[chosen01],
                        X_rest[chosen10],
                        X_rest[chosen11]
                    ], axis=0)
                    f_train = np.concatenate([
                        f_rest[chosen00],
                        f_rest[chosen01],
                        f_rest[chosen10],
                        f_rest[chosen11]
                    ], axis=0)
                    weights_train = np.concatenate([
                        weights_rest[chosen00],
                        weights_rest[chosen01],
                        weights_rest[chosen10],
                        weights_rest[chosen11]
                    ], axis=0)
                else:
                    X_train, f_train, weights_train = [X_rest, f_rest, weights_rest]


                # Reset graph
                tf.keras.backend.clear_session()

                model = FhatModel(p["models_layers_and_parameters"]["fhat"])

                inp_train = dict_to_tensors({"X": X_train, "f": f_train, "weights": weights_train})
                _ = model.train(inp_train)
                final_output = model.run(inp_train, training=False)

                #print("Final loss on training data (could be resampled) is", final_output["loss"].numpy())
                #print("Final accuracy on training_data (could be resampled) is", final_output["accuracy"].numpy())

                inp_rest = dict_to_tensors({"X": X_rest, "f": f_rest, "weights": weights_rest})
                output_rest = model.run(inp_rest, training=False)

                #print("Final accuracy on 4/5 data (not resampled) is", output_rest["accuracy"].numpy())

                inp_kth = dict_to_tensors({"X": X_kth, "f": f_kth, "weights": weights_kth})
                output_kth = model.run(inp_kth, training=False)

                #print("Final accuracy on kth fold (not resampled, we never resample this) is", output_kth["accuracy"].numpy())

                for variable_name in fhats.keys():
                    fhats[variable_name][lower:upper] = output_kth[variable_name]


                indexes00 = f_kth[:, 0] == 1
                indexes01 = f_kth[:, 1] == 1
                indexes10 = f_kth[:, 2] == 1
                indexes11 = f_kth[:, 3] == 1

                inp_00 = dict_to_tensors({"X": X_kth[indexes00], "f": f_kth[indexes00], "weights": weights_kth[indexes00]})
                output_00 = model.run(inp_00, training=False)

                #print("Final accuracy on kth fold samples with y_1 = 0 y_2 = 0: ", output_00["accuracy"].numpy())

                inp_01 = dict_to_tensors({"X": X_kth[indexes01], "f": f_kth[indexes01], "weights": weights_kth[indexes01]})
                output_01 = model.run(inp_01, training=False)

                #print("Final accuracy on kth fold samples with y_1 = 0 y_2 = 1: ", output_01["accuracy"].numpy())

                inp_10 = dict_to_tensors({"X": X_kth[indexes10], "f": f_kth[indexes10], "weights": weights_kth[indexes10]})
                output_10 = model.run(inp_10, training=False)

                #print("Final accuracy on kth fold samples with y_1 = 1 y_2 = 0: ", output_10["accuracy"].numpy())

                inp_11 = dict_to_tensors({"X": X_kth[indexes11], "f": f_kth[indexes11], "weights": weights_kth[indexes11]})
                output_11 = model.run(inp_11, training=False)

                #print("Final accuracy on kth fold samples with y_1 = 1 y_2 = 1: ", output_11["accuracy"].numpy())


            # We exited, save fhats now
            fhat00 = fhats["fhat00"]
            fhat01 = fhats["fhat01"]
            fhat10 = fhats["fhat10"]
            fhat11 = fhats["fhat11"]
            fhat00_calculated = np.copy(fhat00)
            fhat01_calculated = np.copy(fhat01)
            fhat10_calculated = np.copy(fhat10)
            fhat11_calculated = np.copy(fhat11)



        rho_analysis_results =         rho_analysis(X_s, y_1_s, y_2_s, f00_s, f01_s, f10_s, f11_s, fhat00, fhat01,                     fhat10, fhat11, weights_s, weights_original_s, p["mu_4_choice"])

        rho_loop = rho_analysis_results["rho_loop"]
        I = rho_analysis_results["I"]
        J = rho_analysis_results["J"]

        rho_list[s] = rho_loop
        I_list[s] = I
        J_list[s] = J
        sigma_squared = I / (samples * J**2)
        sigma_list[s] = math.sqrt(sigma_squared)
        T = rho_loop / math.sqrt(sigma_squared)
        T_list[s] = T

        #print("Resulting rho:", rho_loop)

        if s == 0 and p["power_test_or_bootstrap"]=="power_test":

            phat = rho_analysis_results["phat"]
            qhat = rho_analysis_results["qhat"]

            a = phat*qhat / ((1-phat)*(1-qhat)+p["eps"])
            b = phat*(1-qhat)/(qhat*(1-phat)+p["eps"])
            #a = phat*qhat / np.maximum((1-phat)*(1-qhat), 0.00001)
            #b = phat*(1-qhat)/np.maximum(qhat*(1-phat), 0.00001)

            L = -np.sqrt(np.minimum(a, 1/a))
            U =  np.sqrt(np.minimum(b, 1/b))

            maxL = np.amax(L)
            minU = np.amin(U)
            #RO_TEST = min(max(maxL, RO_TEST), minU)

            #gen_f11 = RO_TEST*np.sqrt(phat*(1-phat)*qhat*(1-qhat))+phat*qhat
            if p["negative_samples"] == "delete":
                gen_f11 = p["rho_test"]*np.sqrt(np.maximum(0.0, phat*(1-phat)*qhat*(1-qhat)))+phat*qhat
                gen_f01 = qhat-gen_f11
                gen_f10 = phat-gen_f11
                gen_f00 = 1.0 - gen_f01 - gen_f10 - gen_f11

                gen_f11_neg = np.where(gen_f11 < 0.0)[0]
                gen_f01_neg = np.where(gen_f01 < 0.0)[0]
                gen_f10_neg = np.where(gen_f10 < 0.0)[0]
                gen_f00_neg = np.where(gen_f00 < 0.0)[0]

                gen_f_neg_list = np.union1d(
                    np.union1d(gen_f11_neg, gen_f01_neg), \
                    np.union1d(gen_f10_neg, gen_f00_neg), \
                )

                # Remove negative samples
                gen_f11_new = np.delete(gen_f11, gen_f_neg_list, axis=0)
                gen_f01_new = np.delete(gen_f01, gen_f_neg_list, axis=0)
                gen_f10_new = np.delete(gen_f10, gen_f_neg_list, axis=0)
                gen_f00_new = np.delete(gen_f00, gen_f_neg_list, axis=0)


                samples = samples - gen_f_neg_list.shape[0]
                samples_per_fold = math.floor(samples/p["k"])
                samples = samples_per_fold*p["k"]
                bounds = list(map(lambda x: (x, x+samples_per_fold), list(range(0, samples_per_fold*p["k"], samples_per_fold))))

                # Adjust X and weights
                X = np.delete(X, gen_f_neg_list, axis=0)[:samples]
                weights = np.delete(weights, gen_f_neg_list, axis=0)[:samples]
                weights_original = np.delete(weights_original, gen_f_neg_list, axis=0)[:samples]


            elif p["negative_samples"] == "set_to_zero": 
                gen_f11 = p["rho_test"]*np.sqrt(np.maximum(0.0, phat*(1-phat)*qhat*(1-qhat)))+phat*qhat
                gen_f11_neg = np.where(gen_f11 < 0.0)[0]
                gen_f11[gen_f11 < 0.0] = 0.0

                gen_f01 = qhat-gen_f11
                gen_f01_neg = np.where(gen_f01 < 0.0)[0]
                gen_f01[gen_f01 < 0.0] = 0.0

                gen_f10 = phat-gen_f11
                gen_f10_neg = np.where(gen_f10 < 0.0)[0]
                gen_f10[gen_f10 < 0.0] = 0.0

                gen_f00 = 1.0 - gen_f01 - gen_f10 - gen_f11
                gen_f00_neg = np.where(gen_f00 < 0.0)[0]
                gen_f00[gen_f00 < 0.0] = 0.0

                gen_f_neg_list = np.union1d(
                    np.union1d(gen_f11_neg, gen_f01_neg), \
                    np.union1d(gen_f10_neg, gen_f00_neg), \
                )
            else:
                raise Exception("Did not understand NEGATIVE_SAMPLES")


            gen_f = np.concatenate([gen_f00_new, gen_f01_new, gen_f10_new, gen_f11_new], axis=1)
            gen_f = gen_f / dim(np.sum(gen_f, axis=1), 1)

            # Replace
            f = np.zeros((samples, 4))
            y_1 = np.zeros((samples, 1))
            y_2 = np.zeros((samples, 1))
            for i in range(samples):
                fsi = np.random.choice(a=[0, 1, 2, 3], p=gen_f[i])
                f[i, fsi] = 1
                if fsi == 2 or fsi == 3:
                    y_1[i] = 1
                if fsi == 1 or fsi == 3:
                    y_2[i] = 1


    # In[16]:


    if p["power_test_or_bootstrap"]=="power_test":
        #print("Power test: ro for the test is set at", p["rho_test"])

        ro_s_0 = rho_list[0]
        T_s_0 = T_list[0]

        ro_list = rho_list[1:]
        sigma_list = sigma_list[1:]
        T_list = T_list[1:]
        I_list = I_list[1:]
        J_list = J_list[1:]

        #print("maxL:", maxL)
        #print("minU:", minU)

        # Some extra stuff I put here
        #print("Number of samples we had to remove/set-to-zero because < 0:", gen_f_neg_list.shape[0])

        # Append
        accumulated_data_append = {
            "T_list": T_list,
            "I_list": I_list,
            "J_list": J_list,
            "negative_samples": gen_f_neg_list.shape[0],
            "ro_list": ro_list,
            "sigma_list": sigma_list,
            "rho_test": p["rho_test"],
            "mu_4_choice": p["mu_4_choice"],
            "ro_s_0": ro_s_0,
            "T_s_0": T_s_0,
        }
        try:
            accumulated_data = np.load(OUTPUT_FOLDER+ACCUMULATED_CONSTANT_RO_DATA_PATH, allow_pickle=True) 
            accumulated_data = list(accumulated_data)
        except:
            accumulated_data = []
        np.save(OUTPUT_FOLDER+ACCUMULATED_CONSTANT_RO_DATA_PATH, accumulated_data+[accumulated_data_append])



    # Save the parameters
    #for model_name in p["models_layers_and_parameters"].keys():
    #    p["models_layers_and_parameters"][model_name]["make_nn_layers"] =         p["models_layers_and_parameters"][model_name]["make_nn_layers"]().to_json()
     #   p["models_layers_and_parameters"][model_name]["optimizer"] =         tf.keras.optimizers.serialize(p["models_layers_and_parameters"][model_name]["optimizer"])

    with open(OUTPUT_FOLDER+PARAMETERS+".pkl", 'wb') as handle:
        pickle.dump(p, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("All best ro found (first is original data):", rho_list)
    print("All sigma found (first is original data):", sigma_list)
    print("All I found (first is original data):", I_list)
    print("All J found (first is original data):", J_list)
    print("All T found (first is original data):", T_list)
    print(np.sum(rho_list[1:]<rho_list[0]), "out of", rho_list.shape[0]-1, "are less than ww 0")
    print("ro mean:", np.mean(rho_list))
    print("Number of times T < -1.64:", np.sum(T_list[1:] < -1.64) , "( out of", T_list.shape[0]-1, ")")
    print("ro stddev:", np.std(rho_list))



