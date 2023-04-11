from privacy_meter.constants import MetricEnum
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackType
from AttackPipeline import AttackPipeline
from DatasetRepo import *
from Constants import *
from ModelParams import *


"""
Enums for attacks
Fixed input params set for individual attacks

automatic or manual data type enter. --->image or tabular
UMLS diagrams for code understanding



"""

attacks_tf_p = [
                AttackType.THRESHOLD_ATTACK,
                AttackType.LOGISTIC_REGRESSION,
                AttackType.MULTI_LAYERED_PERCEPTRON,
                AttackType.RANDOM_FOREST,
                AttackType.K_NEAREST_NEIGHBORS,
                AttackType.THRESHOLD_ENTROPY_ATTACK
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

dataset_parameters = {
    num_train_points: 5000,
    num_test_points: 5000,
    verbose: 1
}

model_training_params = {
    epoch: 5, 
    batch_size: 64, 
    verbose:2, 
    model_type: ModelType.TensorflowModel,
    loss_fn: tf.keras.losses.CategoricalCrossentropy(),
    optim_fn: 'adam'
    }

attack_parameters = {
    num_population_points: 10000,
    model_type: ModelType.TensorflowModel,
    fpr_tolerance_list: [
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
        ]
}


attack_pipeline_population = AttackPipeline(
    RegisteredDataset.CIFAR100,MetricEnum.POPULATION ,dataset_parameters
    )

model = attack_pipeline_population.get_model(cifar_100_model_2, model_training_params)

attack_pipeline_population.run_attack(model, attack_parameters)

