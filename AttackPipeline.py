from DatasetRepo import *
from ModelParams import *
import tensorflow as tf
from scipy import special
from AttackInputs import AttackInputs
import numpy as np
from Constants import *
import pickle
import enum
from sklearn.metrics import accuracy_score


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



class AttacksAvailable(enum.Enum):
    AttackType.THRESHOLD_ATTACK,
    AttackType.LOGISTIC_REGRESSION,
    AttackType.MULTI_LAYERED_PERCEPTRON,
    AttackType.RANDOM_FOREST,
    AttackType.K_NEAREST_NEIGHBORS,
    AttackType.THRESHOLD_ENTROPY_ATTACK,
    MetricEnum.POPULATION,
    MetricEnum.REFERENCE,
    MetricEnum.SHADOW


attacks_tf_p = [
                AttackType.THRESHOLD_ATTACK,
                AttackType.LOGISTIC_REGRESSION,
                AttackType.MULTI_LAYERED_PERCEPTRON,
                AttackType.RANDOM_FOREST,
                AttackType.K_NEAREST_NEIGHBORS,
                AttackType.THRESHOLD_ENTROPY_ATTACK
            ]


# we can join the attacks
attacks_ml_pr = [
    MetricEnum.POPULATION,
    MetricEnum.REFERENCE,
    MetricEnum.SHADOW
    #, MetricEnum.GROUPPOPULATION
]



class AttackPipeline :




    def __init__(self, dataset_name,attack,dataset_train_params = {} ):
        self.attack_method_dic = {
            AttackType.THRESHOLD_ATTACK : AttackMethod.TF_PRIVACY,
            AttackType.LOGISTIC_REGRESSION : AttackMethod.TF_PRIVACY,
            AttackType.MULTI_LAYERED_PERCEPTRON : AttackMethod.TF_PRIVACY,
            AttackType.RANDOM_FOREST : AttackMethod.TF_PRIVACY,
            AttackType.K_NEAREST_NEIGHBORS : AttackMethod.TF_PRIVACY,
            AttackType.THRESHOLD_ENTROPY_ATTACK : AttackMethod.TF_PRIVACY,
            MetricEnum.POPULATION: AttackMethod.ML_PRIVACY,
            MetricEnum.REFERENCE: AttackMethod.ML_PRIVACY,
            MetricEnum.SHADOW: AttackMethod.ML_PRIVACY
        # , MetricEnum.GROUPPOPULATION
        }
        self.parameter = dataset_train_params
        self.attack = attack
        self.attack_method = self.attack_method_dic[attack]
        print("getting data for training")
        self.getDataSet(dataset_name,self.attack, dataset_train_params)
        print("data gathering succesful")


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
        ignore_keys = ['model_type','loss_fn', fpr_tolerance_list]
        for key in model_training_params.keys() :
            if not key in ignore_keys:
                ans += str(key) + "_" + str(model_training_params[key]).replace(".","")

        return ans + EXTN

    def get_model_file(self, model_name, model_training_params):
        folder = "./wine_quality_models/" #"./purchase_models/" #"./titanic_models/"  # "./adult_income_models/"
        return folder + model_name + self.get_model_param_tr_string(model_training_params)

    def get_model_args(self, model_training_params):

        # Define the keys that correspond to arguments of another_function
        arg_keys = [num_layers_training, num_neigh, hidden_layers,activation,max_depth]

        # Create a dictionary of keyword arguments for another_function
        kwargs = {key: model_training_params[key] for key in arg_keys if key in model_training_params}

        return kwargs

    # make this code more modular, make use of unpacking of function arguments so that the training , attacks, slicing specs , inputs, ml-privacy and tf privacy all can be included !!
    def get_model(self,  model_name, model_training_params):
        model_file = self.get_model_file(model_name, model_training_params)
        try :
            # return if the model with given params is already trained before
            model = self.load_model(model_file)
            # try :
            #     print("[-------------****-----------")
            #     print(model_file)
            #     hist = model.fit(self.x_train, self.y_train)
            #     y_pred = model.predict(self.x_val)
            #     accuracy = accuracy_score(self.y_val, y_pred)
            #     print("Target model val Accuracy:", accuracy)
            #     self.val_acc = accuracy
            #     y_pred = model.predict(self.x_train)
            #     accuracy = accuracy_score(self.y_train, y_pred)
            #     print("Target model target Accuracy:", accuracy)
            #     self.tr_acc = accuracy
            # except:
            #     print("In except block")

            return  model
        except :
            model_func = getattr(ModelParams, model_name)
            if(model_training_params[model_hyper_param]) :
                # model = model_func(ModelParams(), layers_to_freeze = model_training_params[num_layers_training], num_classes = model_training_params[num_class])
                kwargs = self.get_model_args(model_training_params)
                model = model_func(ModelParams(), **kwargs)
            else :
                model = model_func(ModelParams())
            if self.attack == MetricEnum.REFERENCE:
                x = self.datasets_list[0].get_feature('train', '<default_input>')
                y = self.datasets_list[0].get_feature('train', '<default_output>')
                (self.x_train, self.y_train), (self.x_val, self.y_val) = (x,y),None
            elif self.attack == MetricEnum.POPULATION:
                (self.x_train, self.y_train), (self.x_val, self.y_val) = self.dataset.get_data_for_training(model_training_params)
            else:
                (self.x_train, self.y_train), (self.x_val, self.y_val) = self.dataset.get_data_for_training(model_training_params)
            # training the
            # model.compile(optimizer=model_training_params[optim_fn], loss=model_training_params[loss_fn], metrics=['accuracy'])
            try :
                hist = model.fit(self.x_train, self.y_train,
                                 epochs=model_training_params[epoch],
                                 batch_size=model_training_params[batch_size],
                                 verbose=model_training_params[verbose],
                                 validation_data=(self.x_val, self.y_val))
                try :
                    y_pred = model.predict(self.x_val)
                    accuracy = accuracy_score(self.y_val, y_pred)
                    print("Target model val Accuracy:", accuracy)
                    self.val_acc = accuracy
                except :
                    print("")

            except : # for sk learn algos
                hist = model.fit(self.x_train, self.y_train)
                y_pred = model.predict(self.x_val)
                accuracy = accuracy_score(self.y_val, y_pred)
                print("------------------*****---------------")
                print(model_file)
                print("Target model val Accuracy:", accuracy)
                self.val_acc = accuracy
                y_pred = model.predict(self.x_train)
                accuracy = accuracy_score(self.y_train, y_pred)
                print("Target model target Accuracy:", accuracy)
                self.tr_acc = accuracy
                print("-------------****-----------]")

            self.save_model(model, model_file)

            return model

    def getDataSet(self, dataset_name,attack,dataset_train_params ):
        self.dataset = DatasetRepo(dataset_name, dataset_train_params)  #
        (self.x_train_all, self.y_train_all), (self.x_val_all, self.y_val_all) = self.dataset.loadData(dataset_train_params)
        # self.Dataset_ready(self,target_model, dataset_train_params)


        return self.dataset.get_data_for_training()

    def split_data_for_population_attack(self, attack_params = None):
        # l_tr = len(self.x_train_all)
        # l_te = len(self.x_val_all)
        # if( dataset_params['num_train_points'] == None) :
        #     num_train_points = l_tr/2
        # if( dataset_params['num_test_points'] == None) :
        #     num_test_points = l_te/2
        # if(dataset_params["num_population_points"] == None ) :
        #     num_population_points = num_train_points + num_test_points
        # self.x_train, self.y_train = self.x_train_all[:num_train_points], self.y_train_all[:num_train_points]
        # self.x_test, self.y_test = self.x_val_all[:num_test_points], self.y_val_all[:num_test_points]
        self.x_population = self.x_train_all[self.parameter[num_train_points]:(self.parameter[num_train_points] + attack_params[num_population_points])]
        self.y_population = self.y_train_all[self.parameter[num_train_points]:(self.parameter[num_train_points] + attack_params[num_population_points])]

    def split_data_for_reference_attack(self):
        self.x_train, self.y_train = self.x_train_all, self.y_train_all
        self.x_test, self.y_test = self.x_val_all, self.y_val_all
        pass




    def Dataset_ready(self,target_model, attack_params):

        if self.attack == MetricEnum.POPULATION:
            self.split_data_for_population_attack(attack_params)
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

        if self.attack == reference:
            self.split_data_for_reference_attack()
            dataset = Dataset(
                data_dict={
                    'train': {'x': self.x_train, 'y': self.y_train},
                    'test': {'x': self.x_test, 'y': self.y_test}
                },
                default_input='x',
                default_output='y'
            )
            self.datasets_list = dataset.subdivide(
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
                    self.datasets_list[model_idx + 1].get_feature('train', '<default_input>'),
                    self.datasets_list[model_idx + 1].get_feature('train', '<default_output>'),
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2
                )
                reference_models.append(
                    TensorflowModel(model_obj=reference_model, loss_fn=loss_fn)
                )
            target_info_source = InformationSource(
                models=[target_model],
                datasets=[self.datasets_list[0]]
            )

            reference_info_source = InformationSource(
                models=reference_models,
                datasets=[self.datasets_list[0]] # we use the same dataset for the reference models
            )
        return target_info_source,reference_info_source


    def run_attack(self, model, attack_parameters, model_file_name = None ):



        # predictions on the model

        # Prepare inputs for the mia attacks
        if(self.attack_method == AttackMethod.TF_PRIVACY) :
            self.perform_tf_privacy_attack(model , attack_parameters, model_file_name)

        elif(self.attack_method == AttackMethod.ML_PRIVACY) :
            self.perform_ml_privacy_attack(model , attack_parameters)


    def perform_ml_privacy_attack(self, model , attack_parameters):

        # get appropriate model class
        self.model = model
        target_model = None
        if attack_parameters[model_type] == ModelType.TensorflowModel :
            target_model = TensorflowModel(model_obj=model, loss_fn=attack_parameters[loss_fn])
        self.attack_input_params = attack_parameters
        target_info_source, reference_info_source = self.Dataset_ready(target_model, attack_parameters)



        if (self.attack == MetricEnum.POPULATION):
            self.audit_obj = Audit(
                metrics=MetricEnum.POPULATION,
                inference_game_type=InferenceGame.PRIVACY_LOSS_MODEL,
                target_info_sources=target_info_source,
                reference_info_sources=reference_info_source,
                fpr_tolerances=attack_parameters[fpr_tolerance_list]
            )
            self.metrics(audit_obj=self.audit_obj, verbose=True)

        if (self.attack == reference):
            self.audit_obj = Audit(
                metrics=MetricEnum.REFERENCE,
                inference_game_type=InferenceGame.PRIVACY_LOSS_MODEL,
                target_info_sources=target_info_source,
                reference_info_sources=reference_info_source,
                fpr_tolerances=attack_parameters[fpr_tolerance_list]
            )
            self.metrics(audit_obj=self.audit_obj, verbose=True)


    def metrics(self, audit_obj , verbose=False):
        audit_obj.prepare()
        audit_results = audit_obj.run()[0]

        if verbose == True:
            for result in audit_results:
                print(result)

    def get_logits(self, x_train, x_val, model ,bt_size =None ):
        try :
            print('Predict on train...')
            logits_train = model.predict(x_train, batch_size=bt_size)
            print('Predict on test...')
            logits_test = model.predict(x_val, batch_size=bt_size)
        except :
            print('Predict on train...')
            logits_train = model.predict(x_train )
            print('Predict on test...')
            logits_test = model.predict(x_val )

        (logits_train, logits_test) = (np.array(logits_train).reshape(-1,1), np.array(logits_test).reshape(-1,1))

        return (logits_train, logits_test)

    def get_probabilities_sftmax(self, logits_train,logits_test ):
        try:
            prob_train = special.softmax(logits_train, axis=1)
            prob_test = special.softmax(logits_test, axis=1)
        except :
            prob_train = special.softmax(logits_train)
            prob_test = special.softmax(logits_test)

        return (prob_train, prob_test)



    def perform_tf_privacy_attack(self, model , attack_parameters, model_file_name):

        (x_train, y_train), (x_val, y_val) = self.dataset.get_data_for_training()
        bt_size = attack_parameters[batch_size]

        (logits_train, logits_test) = self.get_logits( x_train, x_val, model ,bt_size )

        print('Apply softmax to get probabilities from logits...')
        (prob_train, prob_test) = self.get_probabilities_sftmax(logits_train,logits_test )


        print('Compute losses...')
        cce = tf.keras.backend.categorical_crossentropy
        constant = tf.keras.backend.constant
        try :
            loss_train = cce(constant(y_train.numpy()), constant(prob_train), from_logits=False).numpy()
            loss_test = cce(constant(y_val.numpy()), constant(prob_test), from_logits=False).numpy()
            labels_train = np.argmax(y_train, axis=1)
            labels_test = np.argmax(y_val, axis=1)
        except :
            try :
                y_train_rs = y_train.reshape((y_train.shape[0],1))
                y_val_rs = y_val.reshape((y_val.shape[0], 1))
                labels_train = y_train
                labels_test = y_val
            except :
                y_train_rs = np.argmax(y_train, axis=1)#y_train.to_numpy().reshape(-1,1)
                y_val_rs = np.argmax(y_val, axis=1) #y_val.to_numpy().reshape(-1,1)
                labels_train = y_train #.to_numpy()
                labels_test = y_val #.to_numpy()
            prob_train = prob_train.reshape(prob_train.shape)
            prob_test = prob_test.reshape(prob_test.shape)
            loss_train = cce(constant(y_train), constant(prob_train), from_logits=False).numpy() # _rs
            loss_test = cce(constant(y_val), constant(prob_test), from_logits=False).numpy() # _rs



        attack_inputs = AttackInputs()

        input = attack_inputs.get_Attack_inputs(logits_train, logits_test, loss_train, loss_test, labels_train,
                                                labels_test)

        slicing_specs = attack_inputs.get_Attack_slicing_specs(True, True, True)  # pass values using unpack function arguments!!

        attacks_result = mia.run_attacks(input,
                                         slicing_specs,
                                         attack_types=[self.attack])

        # Plot the ROC curve of the best classifier
        fig = plotting.plot_roc_curve(
            attacks_result.get_result_with_max_auc().roc_curve)

        # Print a user-friendly summary of the attacks
        print(attacks_result.summary(by_slices=True))
        self.write_results(model_file_name, model, attack_parameters, attacks_result)

    def write_results(self, model_file_name, model, attack_parameters, attacks_result):

        # try:
        #     y_pred = model.predict(self.x_val)
        #     val_Acc = accuracy_score(self.y_val, y_pred)
        #     print("Target model val Accuracy:", val_Acc)
        #     y_pred = model.predict(self.x_train)
        #     target_Acc = accuracy_score(self.y_train, y_pred)
        #     print("Target model training Accuracy:", target_Acc)
        # except:
        #     y_pred = model.predict(self.x_val)
        #     val_Acc = accuracy_score(self.y_val, y_pred)
        #     print("Target model val Accuracy:", val_Acc)
        #     y_pred = model.predict(self.x_train)
        #     target_Acc = accuracy_score(self.y_train, y_pred)
        #     print("Target model training Accuracy:", target_Acc)

        try:
            summary_file = summary + TXT_EXTN
            with open(summary_file, encoding="utf-8", mode='a') as f:
                f.write(f"\n------------***************\nNew Summary : {model_file_name}\n")
                line = f" \nsummary : {attacks_result.summary(by_slices=True)} \n***************------------\n"

                f.write(line)
                # line2 = f"\nTarget_model_Accuracies: training {self.tr_acc} , val_acc: {self.val_acc} \n"
                # f.write(line2)
                print(f"\n------------***************\n New Summary :> {model_file_name}\n")
                print(line)
                # print(line2)
                # print("***************------------")
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