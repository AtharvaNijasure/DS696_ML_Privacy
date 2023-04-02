import numpy as np
import tensorflow as tf

from privacy_meter.audit import Audit, MetricEnum
from privacy_meter.audit_report import ROCCurveReport, SignalHistogramReport
from privacy_meter.constants import InferenceGame
from privacy_meter.dataset import Dataset
from privacy_meter.information_source import InformationSource
from privacy_meter.model import TensorflowModel

from privacy_meter.metric import PopulationMetric
from privacy_meter.information_source_signal import ModelGradientNorm, ModelGradient, ModelLoss
from privacy_meter.hypothesis_test import linear_itp_threshold_func
from privacy_meter.model import PytorchModelTensor

"""
 
tested model hyperparameters:

    we use a special class called constants that will have the hyperparemeters being tested
    this is called set_hp()

tested attack parameters:
    we use separate class to control the attack hyper parameters. based on attacks
    
load data:
    for well defined datasets we will do all the necessory processing and make the data ready.

load model:
    the models are completely in the control of the user as there is no fixed model to be used for well known datasets too.







"""

class ML_PM:
    # import numpy as np
    # shape = np.array()
    # train_x = np.array()
    # train_y = np.array()
    """
    define a new init that does until model training completely
    if we have a trained model then we can skip the model trianing
    """
    

    def __init__(self):
        self.set_hp()
        self.set_pm()
        pass


    """
    load the data into the class
    """
    def load_data(self,input,preprocess_x = None, preprocess_y = None):
        
        #only for keras
        (self.x_train_all, self.y_train_all), (self.x_test_all, self.y_test_all) = input
        self.shape = self.x_train_all[0].shape

        if(preprocess_x != None):
            self.x_train_all = preprocess_x(self.x_train)
            self.x_test_all = preprocess_x(self.x_test)

        if(preprocess_y != None):
            self.y_train_all = preprocess_y(self.y_train)
            self.y_test_all = preprocess_y(self.y_test)

        self.x_train, self.y_train = self.x_train_all[:self.num_train_points], self.y_train_all[:self.num_train_points]
        self.x_test, self.y_test = self.x_test_all[:self.num_test_points], self.y_test_all[:self.num_test_points]
        self.x_population = self.x_train_all[self.num_train_points:(self.num_train_points + self.num_population_points)]
        self.y_population = self.y_train_all[self.num_train_points:(self.num_train_points + self.num_population_points)]


    def Dataset_ready(self):
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


    def set_hp(self):
        self.num_train_points = 5000
        self.num_test_points = 5000
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy()
        self.optim_fn = 'adam'
        self.epochs = 25
        self.batch_size = 64
        self.regularizer_penalty = 0.01
        self.regularizer = tf.keras.regularizers.l2(l=self.regularizer_penalty)
        self.verbose = 2

    def set_pm(self):
        self.split_method = "hybrid"
        self.reference_models = 10
        self.num_population_points = 10000
        self.fpr_tolerance_list = [
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
        ]


    """
    change the input to a desired format
    fx - function to change all the datapoints

    """
    def preprocessor_input(self,fx):
        self.x_test = fx(self.x_test)
        self.x_train = fx(self.x_train)
        self.x_population = fx(self.x_population)


    """
    processing the output

    we can make it one_hot or rescale the ouput
    """
    def preprocessor_output(self,fx):
        self.y_test = fx(self.y_test)
        self.y_train = fx(self.y_train)
        self.y_population = fx(self.y_population)

    """
    
    loading model into the class

    """
    def load_model(self,model):
        # IF MODEL IS A string then use default
        # else it must be pytorch or tf model
        self.model = model


    """
    training the model using the hyperparameters of the model

    """
    def train_model(self,verbose = False):
        x = self.target_dataset.get_feature('train', '<default_input>')
        y = self.target_dataset.get_feature('train', '<default_output>')
        if verbose == True:
            self.model.summary()
        self.model.compile(optimizer=self.optim_fn, loss=self.loss_fn, metrics=['accuracy'])
        self.model.fit(x, y,
                         epochs=self.epochs,
                         batch_size=self.batch_size,
                         verbose=self.verbose,
                         validation_data=(self.x_test, self.y_test))
    

    """perform attack"""
    def attacks(self,attack):
        target_model = TensorflowModel(model_obj=self.model, loss_fn=self.loss_fn)

        target_info_source = InformationSource(
            models=[target_model], 
            datasets=[self.target_dataset]
        )

        reference_info_source = InformationSource(
            models=[target_model],
            datasets=[self.reference_dataset]
        )

        if(attack == "population"):
            metric_used = MetricEnum.POPULATION
        if(attack == "reference"):
            metric_used = MetricEnum.REFERENCE
        
        self.audit_obj = Audit(
            metrics=metric_used,
            inference_game_type=InferenceGame.PRIVACY_LOSS_MODEL,
            target_info_sources=target_info_source,
            reference_info_sources=reference_info_source,
            fpr_tolerances=self.fpr_tolerance_list
        )
    

    def metrics(self,verbose = False):
        self.audit_obj.prepare()
        audit_results = self.audit_obj.run()[0]

        if verbose == True:
            for result in audit_results:
                print(result)
        

    

def main():
    mlp = ML_PM()
    mlp.load_data(tf.keras.datasets.cifar10.load_data())
    print("1")
    def pre_x(a):
        return a.astype("float32")/255
    mlp.preprocessor_input(pre_x)
    print("2")
    num_classes = 10
    def pre_y(a):
        return tf.keras.utils.to_categorical(a, num_classes)
    #the below code breaks as there are multiple inputs
    # use lambdas to fix it
    mlp.preprocessor_output(pre_y)
    print("3")
    mlp.Dataset_ready()
    print("data ready")
    hehe = tf.keras.Sequential()
    hehe.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu',
                                     input_shape=mlp.shape, kernel_regularizer=mlp.regularizer))
    hehe.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    hehe.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu',
                                     kernel_regularizer=mlp.regularizer))
    hehe.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    hehe.add(tf.keras.layers.Flatten())
    hehe.add(tf.keras.layers.Dropout(0.5))
    hehe.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    mlp.load_model(hehe)
    mlp.train_model(True)
    print("4")
    mlp.attacks("population")
    print("5")
    mlp.metrics(True)
    print("6")

        



if __name__ == "__main__":
    main()

