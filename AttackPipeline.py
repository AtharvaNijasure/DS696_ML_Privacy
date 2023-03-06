import DatasetRepo
import ModelParams
import tensorflow as tf
from scipy import special
import AttackInputs
import numpy as np


from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import membership_inference_attack as mia
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackInputData
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import SlicingSpec
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackType

import tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.plotting as plotting




class AttackPipeline :


    # make this code more modular, make use of unpacking of function arguments so that the training , attacks, slicing specs , inputs, ml-privacy and tf privacy all can be included !!


    def __init__(self, model_name, dataset_name, attacks ):

        self.model_name = model_name
        self.dataset_name = dataset_name




        self.dataset = DatasetRepo(dataset_name)

        (x_train, y_train), (x_val, y_val) = self.dataset.get_data_for_training()


        model = ModelParams().model_name_to_func(model_name)

        # training the model
        hist = model.fit(x_train, y_train,
                         epochs=5,
                         batch_size=64,
                         verbose=1,
                         validation_data=(x_val, y_val))

        # predictions on the model

        print('Predict on train...')
        logits_train = model.predict(x_train, batch_size=64)
        print('Predict on test...')
        logits_test = model.predict(x_val, batch_size=64)

        print('Apply softmax to get probabilities from logits...')
        prob_train = special.softmax(logits_train, axis=1)
        prob_test = special.softmax(logits_test, axis=1)

        print('Compute losses...')
        cce = tf.keras.backend.categorical_crossentropy
        constant = tf.keras.backend.constant

        loss_train = cce(constant(y_train.numpy()), constant(prob_train), from_logits=False).numpy()
        loss_test = cce(constant(y_val.numpy()), constant(prob_test), from_logits=False).numpy()

        labels_train = np.argmax(y_train, axis=1)
        labels_test = np.argmax(y_val, axis=1)

        attack_inputs = AttackInputs()

        input  = attack_inputs.get_Attack_inputs(logits_train,logits_test , loss_train, loss_test,labels_train, labels_test)

        slicing_specs = attack_inputs.get_Attack_slicing_specs(True, True, True)  # pass values using unpack function arguments!!

        attacks_result = mia.run_attacks(input,
                                         slicing_specs,
                                         attack_types=attacks)

        # Plot the ROC curve of the best classifier
        fig = plotting.plot_roc_curve(
            attacks_result.get_result_with_max_auc().roc_curve)

        # Print a user-friendly summary of the attacks
        print(attacks_result.summary(by_slices=True))

    # we have a model with pre deifned params ... even with training loss , etc
    # we train the model

    # we have attacks array


    # run attacks in the array using ml-privacy-meter for now

    # store the results


'''
Part 1:
For each of the datasets in 
Image 
Tabular 
Text 
we need functions to 
1. get the data 
2. read / load the data 
3. Format the data if required

4. creating the labels or the outputs 

Get all of the above things done

Note : once a dataset is registered here then just by giving the format, splitting parameters we will not need to make any changes with code here 

Part 2:

Independently define and compile the models 
You can write them in the model params class and call them as required 
- Register (define) the basic architectures of the models 
- for tweaking the other training / optimizing/ epoch / etc  params either create a separate function or put enitire model in a different class and then change its params 
- remember model and its params must have some unique name so that we can track it against the recorded results. 

Note : this is the iteration part so try to keep the code more readable and developer friendly!!


Part 4: (AttackPipeline)

Orchestrator Get required data , get models , 
1. Train the required models using the data 
2. save the model params if required 
3. Get predictions on the models 
4. Prepare Attack inputs based upon tf-privacy/ ml-privacy-meter (call Part 3)
5. Run the attacks 

Part 3: (AttackInputs)
1. prepare inputs based upon model, dataset, and tf_privacy or ml_privacy_meter






'''

