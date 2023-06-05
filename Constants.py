import enum

# model training input param keys
epoch = "epoch"
batch_size = "batch_size"
verbose = "verbose"
attack = "attack"


num_train_points = "num_train_points"
num_test_points = "num_test_points"
num_points_per_train_split = "num_points_per_train_split"
num_points_per_test_split = "num_points_per_test_split"
loss_fn = "loss_fn"
optim_fn = "optim_fn"
epochs = "epochs"
num_reference_models = "num_reference_models"

regularizer_penalty = "regularizer_penalty"
regularizer = "regularizer"

num_population_points = "num_population_points"
fpr_tolerance_list = "fpr_tolerance_list"
model_type = "model_type"
model_hyper_param = "model_hyper_param"
num_class = "num_class"
num_layers_training = "num_layers_training"
num_neigh = "num_neigh"
hidden_layers = "hidden_layers"
activation = "activation"
max_depth = "max_depth"

# serialized model extns
EXTN = ".sav"
summary = "summary" # WineQuality_LR Titanic_MLP_LR_rerun_tf_privacy_05_15
TXT_EXTN = ".txt"

# defined model names

cifar_100_model_1 = "cifar100_model_1"
cifar_100_model_2 = "cifar100_model_2"
model_basic_MLP_1 = "model_basic_MLP_1"

# ml-privacy-attacks
population = "population"
reference = "reference"

# model folders
class ModelFolder() :
    Texas100 = 'texas100_models'
    Titanic = 'titanic_models'
    WineQuality = 'wine_quality_models'
    Purchase100 = 'purchase100_models'
    AdultIncome = 'adult_income_models'
