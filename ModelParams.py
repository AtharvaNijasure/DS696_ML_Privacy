# from symbol import decorator

import tensorflow as tf
from keras import layers
from sklearn import neural_network, model_selection

from Constants import *
import enum

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model


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
        # tf.keras.layers.Dense(64, activation='relu', input_shape=(46,)), # , input_shape=(X_train.shape[1],)
        tf.keras.layers.Dense(32),
        tf.keras.layers.LeakyReLU(alpha=0.01),
        tf.keras.layers.Dense(1, activation='sigmoid')
        # layers.Dense(1, activation="sigmoid")
        ])

        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
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
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            optimizer='adam',
            metrics=['AUC', 'accuracy']
        )

        return model


    def decisionTreeCLS(self):

        model = DecisionTreeClassifier()
        return model

    def sk_learn_LR(self):
        model = LogisticRegression()


        return model


    def resnet_model(self, layers_to_freeze = None, num_classes = None):

        # Load the pre-trained ResNet50 model
        resnet = ResNet50(include_top=False, input_shape=(32, 32, 3), pooling='avg')

        # Freeze the weights of the pre-trained layers
        i = 0
        for layer in resnet.layers:

            layer.trainable = False
            i+=1
            if (layers_to_freeze == i):
                break

        # Add a dense output layer with 10 units for CIFAR-10 classification
        outputs = Dense(num_classes, activation='softmax')(resnet.output)

        # Create a new model with the pre-trained layers and the new output layer
        model = Model(inputs=resnet.input, outputs=outputs)

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model





