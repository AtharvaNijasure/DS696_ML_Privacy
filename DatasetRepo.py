# Central repo for all datasets
import enum
import tensorflow as tf



class RegisteredDataset(enum.Enum) :

    CIFAR100 = 'cifar100'
    # CIFAR10 = 'cifar10'
    # NG20 = '20ng'
    # MNIST = 'mnist'
    # ADULT_INCOME = 'AdultIncome'
    # TITANIC = 'titanic'
    # PURCHASE100 = 'purchase100'
    # TEXAS100 = 'texas100'
    # MOVIELENS = 'movielens'
    # IMDB = 'imdb'



class DatasetRepo :


    def __init__(self, dataset, params):
        self.dataset = dataset
        # self.dataset = RegisteredDataset.CIFAR100 # dataset # RegisteredDataset.CIFAR100
        self.loadData()
        self.format_as_per_params(params)
        self.format_outputs()



    def get_data_for_training(self):
        return (self.x_train, self.y_train), (self.x_val, self.y_val)


    def format_as_per_params(self, params):

        return

    # create a clause for each new enrolled dataset
    def format_outputs(self):
        if(self.dataset == RegisteredDataset.CIFAR100) :
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



    def loadData(self):
        if self.dataset == RegisteredDataset.CIFAR100 :
            (self.x_train, self.y_train), (self.x_val, self.y_val) = tf.keras.datasets.cifar100.load_data()
            return








