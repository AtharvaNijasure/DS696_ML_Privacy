# from symbol import decorator

import tensorflow as tf
from keras import layers
from sklearn import neural_network, model_selection

from Constants import *
import enum
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier


class ModelType(enum.Enum):
    # specifically for ml-privacy-meter

    Model = "Model" # abstract class

    PytorchModel = "PytorchModel"

    TensorflowModel = "TensorflowModel"

    LanguageModel = "LanguageModel"

    HuggingFaceCausalLanguageModel = "HuggingFaceCausalLanguageModel"

    SkLearn = "SkLearn"



class ModelParams :


    def __init__(self):
        print("")



    def model_name_to_func(self, model_name):
        if(model_name == 'cifar100_model_1') :
            return self.cifar100_model_1()






    def cifar100_model_2(self):
        regularizer_penalty = 0.01
        regularizer = tf.keras.regularizers.l2(l=regularizer_penalty)
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu',
                                        input_shape=(32,32,3), kernel_regularizer=regularizer))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu',
                                        kernel_regularizer=regularizer))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(100, activation='softmax'))
        model.compile(optimizer="adam", loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        return model


    def cifar100_model_1(self):
        # constructing the model

        model = tf.keras.models.Sequential([
            layers.Conv2D(16, (3, 3), activation='relu',
                          input_shape=(32, 32, 3), padding='same'),
            layers.Conv2D(32, (3, 3),
                          activation='relu',
                          padding='same'),
            layers.Conv2D(64, (3, 3),
                          activation='relu',
                          padding='same'),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(128, (3, 3),
                          activation='relu',
                          padding='same'),

            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            layers.Dense(100, activation='softmax')
        ])

        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            metrics=['AUC', 'accuracy']
        )

        return model


    def cifar100_model_2_deep(self):
        # constructing the model

        model = tf.keras.models.Sequential([
            layers.Conv2D(16, (3, 3), activation='relu',
                          input_shape=(32, 32, 3), padding='same'),
            layers.Conv2D(32, (3, 3),
                          activation='relu',
                          padding='same'),
            layers.Conv2D(64, (3, 3),
                          activation='relu',
                          padding='same'),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(128, (3, 3),
                          activation='relu',
                          padding='same'),

            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            layers.Dense(100, activation='softmax')
        ])

        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            optimizer='adam',
            metrics=['AUC', 'accuracy']
        )

        return model


    def model_basic_MLP_1(self):
        model = tf.keras.models.Sequential([
            layers.Input( shape=(46,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(16 , activation='relu'),
            layers.Dense(1, activation="sigmoid")
        ])



        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            optimizer='adam',
            metrics=['AUC', 'accuracy']
        )



        return model

    def model_basic_LR_1_titanic(self):
        model = tf.keras.models.Sequential([
        layers.Input( shape=(46,)),
        # layers.Dense(128, activation='relu'),
        # layers.Dense(64, activation='relu'),
        # layers.Dense(16 , activation='relu'),
        layers.Dense(1, activation="sigmoid")
        ])

        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            optimizer='adam',
            metrics=['AUC', 'accuracy']
        )

        return model

    def model_basic_MLP_Deep_10_Titanic(self):
        model = tf.keras.models.Sequential([
            layers.Input( shape=(46,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(16 , activation='relu'),
            layers.Dense(8, activation='relu'),
            layers.Dense(4, activation='relu'),
            layers.Dense(2, activation='relu'),
            layers.Dense(1, activation="sigmoid")
            ])

        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            optimizer='adam',
            metrics=['AUC', 'accuracy']
        )

        return model

    def decisionTreeCLS(self , max_depth):
        model = DecisionTreeClassifier(max_depth=max_depth)
        return model

    def sk_learn_LR(self):
        model = LogisticRegression()
        return model

    def sk_learn_MLP(self, hidden_layers ,lr = None, activation = None, batch_size = None ):
        if(lr == None or activation == None or batch_size == None) :
            model = MLPClassifier(hidden_layer_sizes=hidden_layers)
        else :
            model = MLPClassifier(hidden_layer_sizes = hidden_layers, learning_rate= lr, activation=activation, batch_size=batch_size)
        return model


    def sk_learn_KNN(self, num_neigh):
        model = KNeighborsClassifier(n_neighbors= num_neigh)
        return model


    def sk_learn_random_forest(self, depth, random_state = 0, criterion = "gini" ):

        model = RandomForestClassifier(max_depth=depth, random_state= random_state, criterion = criterion)

        return model


