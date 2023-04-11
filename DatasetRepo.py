# Central repo for all datasets
import enum
import tensorflow as tf
import pandas as pd

import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from Constants import *
from privacy_meter.constants import MetricEnum



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
        self.parameter = params
        self.loadData(params)
        self.format_as_per_params(params)
        self.format_outputs()
        # return self.get_data_for_training()



    def get_data_for_training(self,attack = None,model_training_parameters = None):
        if(attack == MetricEnum.POPULATION):
            x_train, y_train = self.x_train[:self.parameter[num_train_points]], self.y_train[:self.parameter[num_train_points]]
            x_test, y_test = self.x_val[:self.parameter[num_test_points]], self.y_val[:self.parameter[num_test_points]]
        else:
            (x_train, y_train),(x_test, y_test) = (self.x_train, self.y_train), (self.x_val, self.y_val)

        return (x_train, y_train),(x_test, y_test)

    


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
            # return (self.x_train, self.y_train), (self.x_val, self.y_val)
        elif self.dataset == RegisteredDataset.CIFAR10 :
            (self.x_train, self.y_train), (self.x_val, self.y_val) = tf.keras.datasets.cifar10.load_data()
            # return

        elif self.dataset == RegisteredDataset.IMDB :
            (self.x_train, self.y_train), (self.x_val, self.y_val) = tf.keras.datasets.imdb.load_data()
            # return

        elif self.dataset == RegisteredDataset.MNIST :
            (self.x_train, self.y_train), (self.x_val, self.y_val) = tf.keras.datasets.mnist.load_data()
            # return

        elif (self.dataset == RegisteredDataset.TITANIC):
            rel_path = "./datasets/titanic/"
            df_tr = pd.read_csv(rel_path + "train.csv")
            df_test = pd.read_csv(rel_path + "test.csv")
            df_test_val = pd.read_csv(rel_path + "gender_submission.csv")
            df_test = pd.merge(df_test, df_test_val)# df_test.join(df_test_val, on = "PassengerId" )

            df = df_tr.append(df_test, ignore_index=True)

            df['Title'] = df.Name.map(lambda x: x.split(',')[1].split('.')[0].strip())

            # inspect the amount of people for each title
            df['Title'].value_counts()

            df = pd.concat([df, pd.get_dummies(df['Title'])], axis=1).drop(labels=['Name'], axis=1)

            df = df.drop(labels=['Cabin','Ticket'], axis=1)



            df = pd.get_dummies(df, columns=['Title','Sex', 'Embarked'])

            # notice that instead of using Title, we should use its corresponding dummy variables
            df_sub = df[['Age', 'Master', 'Miss', 'Mr', 'Mrs', 'SibSp']]

            X_train = df_sub.dropna().drop('Age', axis=1)
            y_train = df['Age'].dropna()
            X_test = df_sub.loc[np.isnan(df.Age)].drop('Age', axis=1)

            regressor = RandomForestRegressor(n_estimators=300)
            regressor.fit(X_train, y_train)
            y_pred = np.round(regressor.predict(X_test), 1)
            df.Age.loc[df.Age.isnull()] = y_pred

            # df.Age.isnull().sum(axis=0)  # no more NAN now


            train_size = int(params["train_size"] * len(df['PassengerId']) )

            df.fillna(value=-1, inplace=True)


            self.y_train = df[0:train_size]['Survived'].values
            self.x_train = df[0:train_size].drop(['Survived', 'PassengerId'], axis=1).values
            self.y_val = df[train_size:]["Survived"].values
            self.x_val = df[train_size:].drop(['Survived', 'PassengerId'], axis=1).values








            # self.x_train = df_tr[["Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"]]
            # self.y_train = df_tr["Survived"]
            # self.x_val = df_test[[ "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin",
            #      "Embarked"]]
            # self.y_val = df_test["Survived"]
            # PassengerId




            # return

        elif (self.dataset == RegisteredDataset.ADULT_INCOME):
            rel_path = "./datasets/adultincome/"
            df = pd.read_csv(rel_path + "adult.csv")
            input_cols = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num',
                       'marital-status', 'occupation', 'relationship', 'race', 'gender',
                       'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']


            self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(df[input_cols], df["income"],
                                                                                  train_size=params["train_size"])

            # return

        elif (self.dataset == RegisteredDataset.PURCHASE100):
            rel_path = "./datasets/purchase100/"
            data = np.load(rel_path + 'purchase100.npz')
            features = data['features']
            labels = data['labels']

            self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(features, labels, train_size = params["train_size"] )

            # note the labels are one hot
            # return
        
        return (self.x_train, self.y_train), (self.x_val, self.y_val)









