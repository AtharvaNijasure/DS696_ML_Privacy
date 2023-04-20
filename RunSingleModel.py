from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackType
from AttackPipeline import AttackPipeline
from DatasetRepo import *
from Constants import *
from ModelParams import *

attacks_tf_p = [
                AttackType.THRESHOLD_ATTACK,
                AttackType.LOGISTIC_REGRESSION,
                AttackType.MULTI_LAYERED_PERCEPTRON,
                AttackType.RANDOM_FOREST,
                AttackType.K_NEAREST_NEIGHBORS
                # , AttackType.THRESHOLD_ENTROPY_ATTACK
            ]


# we can join the attacks
attacks_ml_pr = [
    MetricEnum.POPULATION,
    MetricEnum.REFERENCE,
    MetricEnum.SHADOW
    #, MetricEnum.GROUPPOPULATION
]


"""
attack pipeline - init(dataset,attack,model_train_and_dataset_params = {})
get_model(model_fx, training_params)
run_other_models(#only for reference and shadow)
run_attack(target_model,attack_params)
"""



epoch_list = [10 ] #, 30,40,50,60] # 10,20,
batch_sizes =  [32 ] #, 64,128 ] # 256


model_names = ["resnet_model"] #"model_basic_MLP_Deep_10_Titanic"] # , "model_basic_LR_1_titanic",model_basic_MLP_1 , "sk_learn_LR" ]#



bs = 64
nl = 1
ep = 30
mod_name = "resnet_model"
attack_parameters_titanic = {
    batch_size: bs,
    num_population_points: 400,
    fpr_tolerance_list: [
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
    ],
    loss_fn: tf.keras.losses.CategoricalCrossentropy(),
    model_type: ModelType.TensorflowModel
}
dataset_parameters = {
    num_train_points: 200,
    num_test_points: 200,
    verbose: 2,
    "train_size": 0.8
}

model_training_params = {
    epoch: ep,
    batch_size: bs,
    verbose: 1,
    model_type: ModelType.TensorflowModel,
    loss_fn: tf.keras.losses.CategoricalCrossentropy(),
    optim_fn: 'adam',
    model_hyper_param : True,
    num_class : 100,
    num_layers_training : nl
}



print("here 1")

for attack_name in attacks_tf_p :
    print(f"Started attack {attack_name} for {mod_name} for epoch :{ep} batch_size :{bs}")
    attack_pipeline_population = AttackPipeline(
        RegisteredDataset.CIFAR100, attack_name, dataset_parameters
    )
    model = attack_pipeline_population.get_model(mod_name, model_training_params)
    print("here 2")
    model_file = attack_pipeline_population.get_model_file(mod_name, model_training_params)
    print("here 3")
    attack_pipeline_population.run_attack(model, attack_parameters_titanic, model_file_name = model_file)
    print("here 4")


print(f"Done {mod_name}")