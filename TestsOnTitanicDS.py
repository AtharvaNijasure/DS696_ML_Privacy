from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackType
from AttackPipeline import AttackPipeline
from DatasetRepo import *
from Constants import *
from ModelParams import *

attacks_tf_p = [
                # AttackType.THRESHOLD_ATTACK,
                AttackType.LOGISTIC_REGRESSION,
                AttackType.MULTI_LAYERED_PERCEPTRON,
                AttackType.RANDOM_FOREST,
                AttackType.K_NEAREST_NEIGHBORS
                #, AttackType.THRESHOLD_ENTROPY_ATTACK
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
    verbose: 2,
    "train_size" : 0.8
}

model_training_params = {
    epoch: 50,
    batch_size: 64,
    verbose:1,
    model_type: ModelType.TensorflowModel,
    loss_fn: tf.keras.losses.CategoricalCrossentropy(),
    optim_fn: 'adam'
    }


attack_parameters_titanic = {
    batch_size: 64
}
# attack_pipeline = AttackPipeline(
#     RegisteredDataset.TITANIC, AttackType.LOGISTIC_REGRESSION,dataset_parameters
#     )



# wrapper
# model = attack_pipeline.get_model(cifar_100_model_1, model_training_params)
# attack_pipeline.run_attacks(model, attacks_tf_p,model_training_params, AttackMethod.TF_PRIVACY , tf_attack_input_params)
# attack_pipeline.run_attacks(model, attacks_ml_pr,model_training_params, AttackMethod.ML_PRIVACY , ml_pr_attack_input_params)



for att in attacks_tf_p :
    attack_pipeline_population = AttackPipeline(
        RegisteredDataset.TITANIC, att, dataset_parameters
    )
    model = attack_pipeline_population.get_model(model_basic_MLP_1, model_training_params)
    attack_pipeline_population.run_attack(model, attack_parameters_titanic)