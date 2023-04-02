# Central repo for all datasets
import enum
import tensorflow as tf
import pandas as pd

import seaborn as sns
from sklearn.model_selection import train_test_split
import numpy as np



class AttackMethod(enum.Enum) :
    ML_PRIVACY = 'ml_privacy_meter'
    TF_PRIVACY = 'tf_privacy'

class RegisteredDataset(enum.Enum) :

    CIFAR100 = 'cifar100'
    CIFAR10 = 'cifar10'
    MNIST = 'mnist'
    # Tabular
    ADULT_INCOME = 'AdultIncome'
    TITANIC = 'titanic'
    PURCHASE100 = 'purchase100'
    # TEXAS100 = 'texas100'
    # #Text
    # MOVIELENS = 'movielens'
    IMDB = 'imdb'
    # NG20 = '20ng'


class DatasetRepo :


    def __init__(self, dataset, params):
        self.dataset = dataset
        # self.dataset = RegisteredDataset.CIFAR100 # dataset # RegisteredDataset.CIFAR100
        self.loadData(params)
        self.format_as_per_params(params)
        self.format_outputs()
        # return self.get_data_for_training()



    def get_data_for_training(self):
        return (self.x_train, self.y_train), (self.x_val, self.y_val)


    def format_as_per_params(self, params):

        return

    # create a clause for each new enrolled dataset
    def format_outputs(self):
        one_hots = [RegisteredDataset.CIFAR100 , RegisteredDataset.CIFAR10, RegisteredDataset.MNIST]
        if(self.dataset in one_hots) :
            # preparing the outputs / labels
            self.y_train = tf.one_hot(self.y_train,
                                 depth=self.y_train.max() + 1,
                                 dtype=tf.float64)
            self.y_val = tf.one_hot(self.y_val,
                               depth=self.y_val.max() + 1,
                               dtype=tf.float64)

            self.y_train = tf.squeeze(self.y_train)
            self.y_val = tf.squeeze(self.y_val)
            return
        elif(self.dataset == RegisteredDataset.TITANIC) :
            rel_path = "./datasets/titanic/"



            return



    def loadData(self, params):
        if self.dataset == RegisteredDataset.CIFAR100 :
            (self.x_train, self.y_train), (self.x_val, self.y_val) = tf.keras.datasets.cifar100.load_data()
            return
        elif self.dataset == RegisteredDataset.CIFAR10 :
            (self.x_train, self.y_train), (self.x_val, self.y_val) = tf.keras.datasets.cifar10.load_data()
            return

        elif self.dataset == RegisteredDataset.IMDB :
            (self.x_train, self.y_train), (self.x_val, self.y_val) = tf.keras.datasets.imdb.load_data()
            return

        elif self.dataset == RegisteredDataset.MNIST :
            (self.x_train, self.y_train), (self.x_val, self.y_val) = tf.keras.datasets.mnist.load_data()
            return

        elif (self.dataset == RegisteredDataset.TITANIC):
            rel_path = "./datasets/titanic/"
            df_tr = pd.read_csv(rel_path + "train.csv")
            df_test = pd.read_csv(rel_path + "test.csv")
            df_test_val = pd.read_csv(rel_path + "gender_submission.csv")
            self.x_train = df_tr[["PassengerId","Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"]]
            self.y_train = df_tr[["PassengerId","Survived"]]
            self.x_val = df_test[["PassengerId", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin",
                 "Embarked"]]
            self.y_val = df_test_val[["PassengerId","Survived"]]




            return

        elif (self.dataset == RegisteredDataset.ADULT_INCOME):
            rel_path = "./datasets/adultincome/"
            df = pd.read_csv(rel_path + "adult.csv")
            input_cols = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num',
                       'marital-status', 'occupation', 'relationship', 'race', 'gender',
                       'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']


            self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(df[input_cols], df["income"],
                                                                                  train_size=params["train_size"])

            return

        elif (self.dataset == RegisteredDataset.PURCHASE100):
            rel_path = "./datasets/purchase100/"
            data = np.load(rel_path + 'purchase100.npz')
            features = data['features']
            labels = data['labels']

            self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(features, labels, train_size = params["train_size"] )

            # note the labels are one hot
            return









