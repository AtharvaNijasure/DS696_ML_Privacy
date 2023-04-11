# from symbol import decorator

import tensorflow as tf
from keras import layers
from sklearn import neural_network, model_selection

from Constants import *
import enum


class ModelType(enum.Enum):
    # specifically for ml-privacy-meter

    Model = "Model" # abstract class

    PytorchModel = "PytorchModel"

    TensorflowModel = "TensorflowModel"

    LanguageModel = "LanguageModel"

    HuggingFaceCausalLanguageModel = "HuggingFaceCausalLanguageModel"



class ModelParams :


    def __init__(self):
        print("initiating the model")



    def model_name_to_func(self, model_name):
        if(model_name == 'cifar100_model_1') :
            return self.cifar100_model_1()









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
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            optimizer='adam',
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

