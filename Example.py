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
                AttackType.K_NEAREST_NEIGHBORS,
                AttackType.THRESHOLD_ENTROPY_ATTACK
            ]

attacks_ml_pr = [
    population,
    reference
]


format_dataset_params = {"train_size" : 0.8}
model_training_params = {epoch: 5, batch_size: 64, verbose:1, model_type: ModelType.PytorchModel }
tf_attack_input_params = {}

ml_pr_attack_input_params = {
    "population" :
    {num_train_points: 5000,
    num_test_points: 5000,
    loss_fn: tf.keras.losses.CategoricalCrossentropy(),
    optim_fn: 'adam',
    epochs: 2,
    batch_size: 64,
    regularizer_penalty: 0.01,
    verbose: 2,
    num_population_points: 10000,
    fpr_tolerance_list: [
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
        ]},
    "reference":{
    num_train_points: 5000,
    num_test_points: 5000,
    loss_fn: tf.keras.losses.CategoricalCrossentropy(),
    optim_fn: 'adam',
    epochs: 2,
    batch_size: 64,
    regularizer_penalty: 0.01,
    verbose: 2,
    num_population_points: 10000,
    fpr_tolerance_list: [
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
        ]}
}

ml_pr_attack_input_params[regularizer] : tf.keras.regularizers.l2(l= ml_pr_attack_input_params[regularizer_penalty])


attack_pipeline = AttackPipeline(
    RegisteredDataset.TITANIC , format_dataset_params ,
    num_train_points = ml_pr_attack_input_params[num_train_points],
    num_test_points = ml_pr_attack_input_params[num_test_points],
    num_population_points = ml_pr_attack_input_params[num_population_points]
    )

# wrapper
# model = attack_pipeline.get_model(cifar_100_model_1, model_training_params)
# attack_pipeline.run_attacks(model, attacks_tf_p,model_training_params, AttackMethod.TF_PRIVACY , tf_attack_input_params)
# attack_pipeline.run_attacks(model, attacks_ml_pr,model_training_params, AttackMethod.ML_PRIVACY , ml_pr_attack_input_params)

model = attack_pipeline.get_model(model_basic_MLP_1, model_training_params)
attack_pipeline.run_attacks(model, attacks_tf_p,model_training_params, AttackMethod.TF_PRIVACY , tf_attack_input_params)
attack_pipeline.run_attacks(model, attacks_ml_pr,model_training_params, AttackMethod.ML_PRIVACY , ml_pr_attack_input_params)
attack_pipeline.run_attacks(model, attacks_ml_pr,model_training_params, AttackMethod.ML_PRIVACY , ml_pr_attack_input_params)