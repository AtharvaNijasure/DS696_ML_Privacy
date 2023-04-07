from DatasetRepo import *
from ModelParams import *
import tensorflow as tf
from scipy import special
from AttackInputs import AttackInputs
import numpy as np
from Constants import *
import pickle


from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import membership_inference_attack as mia
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackInputData
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import SlicingSpec
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackType

import tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.plotting as plotting


from privacy_meter.constants import InferenceGame
from privacy_meter.dataset import Dataset
from privacy_meter.information_source import InformationSource
from privacy_meter.model import *
from privacy_meter.audit import Audit, MetricEnum









class AttackPipeline :



    def __init__(self, dataset_name,attack,format_dataset_params = [], num_train_points = None, num_test_points = None, num_population_points = None):
        self.getDataSet(dataset_name,attack, format_dataset_params, num_train_points, num_test_points, num_population_points )


    def save_model(self, model, filename):
        # save the model to disk
        # filename = 'finalized_model.sav'
        pickle.dump(model, open(filename, 'wb'))

    def load_model(self, filename):
        # load the model from disk
        loaded_model = pickle.load(open(filename, 'rb'))
        return loaded_model

    def get_model_param_tr_string(self, model_training_params):
        ans = ""
        for key in model_training_params.keys() :
            ans += str(key) + "_" + str(model_training_params[key]).replace(".","")

        return ans + EXTN

    def get_model_file(self, model_name, model_training_params):
        return model_name + self.get_model_param_tr_string(model_training_params)


    # make this code more modular, make use of unpacking of function arguments so that the training , attacks, slicing specs , inputs, ml-privacy and tf privacy all can be included !!
    def get_model(self,  model_name, model_training_params):
        model_file = self.get_model_file(model_name, model_training_params)
        try :
            # return if the model with given params is already trained before
            return  self.load_model(model_file)
        except :
            model_func = getattr(ModelParams, model_name)
            model = model_func(ModelParams())
            (x_train, y_train), (x_val, y_val) = self.dataset.get_data_for_training()
            # training the model
            hist = model.fit(x_train, y_train,
                             epochs=model_training_params[epoch],
                             batch_size=model_training_params[batch_size],
                             verbose=model_training_params[verbose],
                             validation_data=(x_val, y_val))

            self.save_model(model, model_name + EXTN)

            return model

    def getDataSet(self, dataset_name,attack,format_dataset_params , num_train_points = None, num_test_points = None , num_population_points = None):
        self.dataset = DatasetRepo(dataset_name, format_dataset_params)  #
        (self.x_train_all, self.y_train_all), (self.x_val_all, self.y_val_all) = self.dataset.get_data_for_training()

        if(attack == population):
            self.split_data_for_population_attack(num_train_points, num_test_points, num_population_points)
        if(attack == reference):
            self.split_data_for_reference_attack()



        return self.dataset.get_data_for_training()

    def split_data_for_population_attack(self, num_train_points = None, num_test_points = None, num_population_points = None):
        l_tr = len(self.x_train_all)
        l_te = len(self.x_val_all)
        if( num_train_points == None) :
            num_train_points = l_tr/2
        if( num_test_points == None) :
            num_test_points = l_te/2
        if(num_population_points == None or num_population_points > num_train_points + num_test_points) :
            num_population_points = num_train_points + num_test_points
        self.x_train, self.y_train = self.x_train_all[:num_train_points], self.y_train_all[:num_train_points]
        self.x_test, self.y_test = self.x_val_all[:num_test_points], self.y_val_all[:num_test_points]
        self.x_population = self.x_train_all[num_train_points:(num_train_points + num_population_points)]
        self.y_population = self.y_train_all[num_train_points:(num_train_points + num_population_points)]

    def split_data_for_reference_attack(self):
        self.x_train, self.y_train = self.x_train_all, self.y_train_all
        self.x_test, self.y_test = self.x_val_all, self.y_val_all
        pass




    def Dataset_ready(self,attacks,target_model):
        if attacks == population:
            train_ds = {'x': self.x_train, 'y': self.y_train}
            test_ds = {'x': self.x_test, 'y': self.y_test}
            self.target_dataset = Dataset(
                data_dict={'train': train_ds, 'test': test_ds},
                default_input='x', default_output='y'
            )

            # create the reference dataset
            population_ds = {'x': self.x_population, 'y': self.y_population}
            self.reference_dataset = Dataset(
                # this is the default mapping that a Metric will look for
                # in a reference dataset
                data_dict={'train': population_ds},
                default_input='x', default_output='y'
            )
            target_info_source = InformationSource(
                models=[target_model],
                datasets=[self.target_dataset]
            )

            reference_info_source = InformationSource(
                models=[target_model],
                datasets=[self.reference_dataset]
            )

        if attacks == reference:
            dataset = Dataset(
                data_dict={
                    'train': {'x': self.x_train, 'y': self.y_train},
                    'test': {'x': self.x_test, 'y': self.y_test}
                },
                default_input='x',
                default_output='y'
            )
            datasets_list = dataset.subdivide(
                num_splits=self.attack_input_params[num_reference_models] + 1,
                delete_original=True,
                in_place=False,
                return_results=True,
                method='hybrid',
                split_size={'train': self.attack_input_params[num_points_per_train_split], 'test': self.attack_input_params[num_points_per_test_split]}
            )
            reference_models = []
            for model_idx in range(self.attack_input_params[num_reference_models]):
                print(f"Training reference model {model_idx}...")
                reference_model = self.model
                reference_model.compile(optimizer=optim_fn, loss=loss_fn, metrics=['accuracy'])
                reference_model.fit(
                    datasets_list[model_idx + 1].get_feature('train', '<default_input>'),
                    datasets_list[model_idx + 1].get_feature('train', '<default_output>'),
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2
                )
                reference_models.append(
                    TensorflowModel(model_obj=reference_model, loss_fn=loss_fn)
                )
            target_info_source = InformationSource(
                models=[target_model],
                datasets=[datasets_list[0]]
            )

            reference_info_source = InformationSource(
                models=reference_models,
                datasets=[datasets_list[0]] # we use the same dataset for the reference models
            )
        return target_info_source,reference_info_source


    def run_attacks(self, model, attacks, model_training_params, attack_method , attack_input_params ):



        # predictions on the model

        # Prepare inputs for the mia attacks
        if(attack_method == AttackMethod.TF_PRIVACY) :
            self.perform_tf_privacy_attack(model , attacks, model_training_params , attack_input_params)

        elif(attack_method == AttackMethod.ML_PRIVACY) :
            self.perform_ml_privacy_attack(model , attacks, model_training_params , attack_input_params)


    def perform_ml_privacy_attack(self, model , attacks, model_training_params = None , attack_input_params = None):

        # get appropriate model class
        self.model = model
        target_model = None
        if model_training_params[model_type] == ModelType.PytorchModel :
            target_model = TensorflowModel(model_obj=model, loss_fn=attack_input_params[loss_fn])
        self.attack_input_params = attack_input_params
        # dataset ready
        # target_dataset, reference_dataset = self.Dataset_ready()

        # target_info_source = InformationSource(
        #     models=[target_model],
        #     datasets=[target_dataset]
        # )

        # reference_info_source = InformationSource(
        #     models=[target_model],
        #     datasets=[reference_dataset]
        # )
        target_info_source, reference_info_source = self.Dataset_ready(attacks,target_model)


        for attack in attacks :
            if (attack == population):
                self.audit_obj = Audit(
                    metrics=MetricEnum.POPULATION,
                    inference_game_type=InferenceGame.PRIVACY_LOSS_MODEL,
                    target_info_sources=target_info_source,
                    reference_info_sources=reference_info_source,
                    fpr_tolerances=attack_input_params[fpr_tolerance_list]
                )
                self.metrics(audit_obj=self.audit_obj, verbose=True)

            if (attack == reference):
                self.audit_obj = Audit(
                    metrics=MetricEnum.REFERENCE,
                    inference_game_type=InferenceGame.PRIVACY_LOSS_MODEL,
                    target_info_sources=target_info_source,
                    reference_info_sources=reference_info_source,
                    fpr_tolerances=attack_input_params[fpr_tolerance_list]
                )
                self.metrics(audit_obj=self.audit_obj, verbose=True)


    def metrics(self, audit_obj , verbose=False):
        audit_obj.prepare()
        audit_results = audit_obj.run()[0]

        if verbose == True:
            for result in audit_results:
                print(result)


    def perform_tf_privacy_attack(self, model , attacks, model_training_params , attack_input_params = None):

        (x_train, y_train), (x_val, y_val) = self.dataset.get_data_for_training()

        print('Predict on train...')
        logits_train = model.predict(x_train, batch_size=model_training_params[batch_size])
        print('Predict on test...')
        logits_test = model.predict(x_val, batch_size=model_training_params[batch_size])

        print('Apply softmax to get probabilities from logits...')
        prob_train = special.softmax(logits_train, axis=1)
        prob_test = special.softmax(logits_test, axis=1)

        print('Compute losses...')
        cce = tf.keras.backend.categorical_crossentropy
        constant = tf.keras.backend.constant
        try :
            loss_train = cce(constant(y_train.numpy()), constant(prob_train), from_logits=False).numpy()
            loss_test = cce(constant(y_val.numpy()), constant(prob_test), from_logits=False).numpy()
            labels_train = np.argmax(y_train, axis=1)
            labels_test = np.argmax(y_val, axis=1)
        except :
            y_train_rs = y_train.reshape((y_train.shape[0],1))
            y_val_rs = y_val.reshape((y_val.shape[0], 1))
            loss_train = cce(constant(y_train_rs), constant(prob_train), from_logits=False).numpy()
            loss_test = cce(constant(y_val_rs), constant(prob_test), from_logits=False).numpy()

            labels_train = y_train
            labels_test = y_val

        attack_inputs = AttackInputs()

        input = attack_inputs.get_Attack_inputs(logits_train, logits_test, loss_train, loss_test, labels_train,
                                                labels_test)

        slicing_specs = attack_inputs.get_Attack_slicing_specs(True, True,
                                                               True)  # pass values using unpack function arguments!!

        attacks_result = mia.run_attacks(input,
                                         slicing_specs,
                                         attack_types=attacks)

        # Plot the ROC curve of the best classifier
        fig = plotting.plot_roc_curve(
            attacks_result.get_result_with_max_auc().roc_curve)

        # Print a user-friendly summary of the attacks
        print(attacks_result.summary(by_slices=True))
        try:
            summary_file = summary + TXT_EXTN
            with open(summary_file, encoding="utf-8", mode='a') as f:
                f.write("\nNew Summary :\n")
                line = f" model params :{self.get_model_param_tr_string(model_training_params)[:3]} , summary : {attacks_result.summary(by_slices=True)}"
                f.write(line)
                f.close()
        except:
            print("Error : while saving summary in the summary file")








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

-- https://bargavjayaraman.github.io/project/evaluating-dpml/
-- https://github.com/bargavj/EvaluatingDPML
licenses on the git code before sending to Pallika and Virendra


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
0. Get dataset, format if required 
1. Train the required models using the data 
2. save the model params if required 
3. Get predictions on the models 
4. Prepare Attack inputs based upon tf-privacy/ ml-privacy-meter (call Part 3)
5. Run the attacks 

Part 3: (AttackInputs)
1. prepare inputs based upon model, dataset, and tf_privacy or ml_privacy_meter



'''

