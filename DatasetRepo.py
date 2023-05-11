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
from sklearn.datasets import load_wine


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
    WINEQUALITY = 'winequality'


class DatasetRepo :


    def __init__(self, dataset, params):
        self.dataset = dataset
        # self.dataset = RegisteredDataset.CIFAR100 # dataset # RegisteredDataset.CIFAR100
        self.parameter = params
        self.loadData(params)
        self.format_as_per_params(params)
        # self.format_outputs()
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
        one_hots = [RegisteredDataset.CIFAR100 , RegisteredDataset.CIFAR10, RegisteredDataset.MNIST, RegisteredDataset.WINEQUALITY]
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
                       'marital', 'relationship', 'race', 'gender',
                       'capital gain', 'capital loss', 'hours per week'] # 'occupation', 'country'
            # text_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender','native-country']

            # replacing some special character columns names with proper names
            df = df.rename(
                columns={'capital-gain': 'capital gain', 'capital-loss': 'capital loss', 'native-country': 'country',
                         'hours-per-week': 'hours per week', 'marital-status': 'marital'})



            df['country'] = df['country'].replace('?', np.nan)
            df['workclass'] = df['workclass'].replace('?', np.nan)
            df['occupation'] = df['occupation'].replace('?', np.nan)
            # dropping the NaN rows now
            df.dropna(how='any', inplace=True)

            # dropping based on uniquness of data from the dataset
            df = df.drop(
                [ 'occupation', 'country'],
                axis=1)

            df['income'] = df['income'].map({'<=50K': 0, '>50K': 1}).astype(int)

            # gender
            df['gender'] = df['gender'].map({'Male': 0, 'Female': 1}).astype(int)
            # race
            df['race'] = df['race'].map(
                {'Black': 0, 'Asian-Pac-Islander': 1, 'Other': 2, 'White': 3, 'Amer-Indian-Eskimo': 4}).astype(int)
            # marital
            df['marital'] = df['marital'].map(
                {'Married-spouse-absent': 0, 'Widowed': 1, 'Married-civ-spouse': 2, 'Separated': 3, 'Divorced': 4,
                 'Never-married': 5, 'Married-AF-spouse': 6}).astype(int)
            # workclass
            df['workclass'] = df['workclass'].map(
                {'Self-emp-inc': 0, 'State-gov': 1, 'Federal-gov': 2, 'Without-pay': 3, 'Local-gov': 4, 'Private': 5,
                 'Self-emp-not-inc': 6}).astype(int)
            # education
            df['education'] = df['education'].map(
                {'Some-college': 0, 'Preschool': 1, '5th-6th': 2, 'HS-grad': 3, 'Masters': 4, '12th': 5, '7th-8th': 6,
                 'Prof-school': 7, '1st-4th': 8, 'Assoc-acdm': 9, 'Doctorate': 10, '11th': 11, 'Bachelors': 12,
                 '10th': 13, 'Assoc-voc': 14, '9th': 15}).astype(int)
            # occupation
            # df['occupation'] = df['occupation'].map(
            #     {
            #         'Machine - op - inspct': 0,
            #         'Farming - fishing': 1,
            #         'Protective - serv':2,
            #         'Other - service' :3,
            #         'Prof - specialty' :4,
            #         'Craft - repair':5,
            #         'Adm - clerical':6,
            #         'Exec - managerial':7,
            #         'Tech - support':8,
            #         'Sales':9,
            #         'Priv - house - serv':10,
            #         'Transport - moving':11,
            #         'Handlers - cleaners':12,
            #         'Armed - Forces':13
            #     }).astype(int)

            # country
            # df['country'] = df['country'].map(
            #     {'United - States':0, 'Peru':1,'Guatemala':2,'Mexico':3,
            #         'Dominica n -Republic'		:	4	,
            #         'Ireland'		:	5	,
            #         'Germany'		:	6	,
            #         'Philippines'		:	7	,
            #         'Thailand'		:	8	,
            #         'Haiti'		:	9	,
            #         'E l -Salvador'		:	10	,
            #         'Puert o -Rico'		:	11	,
            #         'Vietnam'		:	12	,
            #         'South'		:	13	,
            #         'Columbia'		:	14	,
            #         'Japan'		:	15	,
            #         'India'		:	16	,
            #         'Cambodia'		:	17	,
            #         'Poland'		:	18	,
            #         'Laos'		:	19	,
            #         'England'		:	20	,
            #         'Cuba'		:	21	,
            #         'Taiwan'		:	22	,
            #         'Italy'		:	23	,
            #         'Canada'		:	24	,
            #         'Portugal'		:	25	,
            #         'China'		:	26	,
            #         'Nicaragua'		:	27	,
            #         'Honduras'		:	28	,
            #         'Iran'		:	29	,
            #         'Scotland'		:	30	,
            #         'Jamaica'		:	31	,
            #         'Ecuador'		:	32	,
            #         'Yugoslavia'		:	33	,
            #         'Hungary'		:	34	,
            #         'Hong'		:	35	,
            #         'Greece'		:	36	,
            #         'Trinada d &Tobago'		:	37	,
            #         'Outlyin g -US(Gua m -USV I -etc)'		:	38	,
            #         'France'		:	39	,
            #         'Holan d -Netherlands'		:	40
            #     }).astype(int)


            # relationship
            df['relationship'] = df['relationship'].map(
                {'Not-in-family': 0, 'Wife': 1, 'Other-relative': 2, 'Unmarried': 3, 'Husband': 4,
                 'Own-child': 5}).astype(int)

            # for d in text_cols :
            #     df = pd.get_dummies(df, columns=[d])


            self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(df[input_cols], df["income"],
                                                                                  train_size=params["train_size"])


            # took help from
            # https://towardsdatascience.com/a-beginners-guide-to-data-analysis-machine-learning-with-python-adult-salary-dataset-e5fc028b6f0a

            # return

        elif (self.dataset == RegisteredDataset.PURCHASE100):
            rel_path = "./datasets/purchase100/"
            data = np.load(rel_path + 'purchase100.npz')
            features = data['features']
            labels = data['labels']

            self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(features, labels, train_size = params["train_size"] )



        elif (self.dataset == RegisteredDataset.WINEQUALITY):
            rel_path = "./datasets/wine_quality/"

            redwine = rel_path + "winequality-red.csv"
            whitewine = rel_path + "winequality-white.csv"

            df = pd.read_csv(redwine)
            df_white = pd.read_csv(whitewine)
            df = df.append(df_white, ignore_index=True)

            # Create Classification version of target variable
            df['goodquality'] = [1 if x >= 7 else 0 for x in df['quality']]
            # Separate feature variables and target variable
            X = df.drop(['quality', 'goodquality'], axis=1)
            y = df['goodquality']



            self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(X, y, train_size = params["train_size"] )

            # note the labels are one hot
            # return
        self.format_outputs()
        return (self.x_train, self.y_train), (self.x_val, self.y_val)










