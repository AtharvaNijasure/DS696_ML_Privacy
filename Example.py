from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackType
from AttackPipeline import AttackPipeline
from DatasetRepo import *
from Constants import *
from ModelParams import ModelParams


attacks = [
                AttackType.THRESHOLD_ATTACK,
                AttackType.LOGISTIC_REGRESSION,
                AttackType.MULTI_LAYERED_PERCEPTRON,
                AttackType.RANDOM_FOREST,
                AttackType.K_NEAREST_NEIGHBORS,
                AttackType.THRESHOLD_ENTROPY_ATTACK
            ]



format_dataset_params = {}
model_training_params = {epoch : 5, batch_size : 64, verbose :1  }
tf_attack_input_params = {}
ml_pr_attack_input_params = {}



attack_pipeline = AttackPipeline( RegisteredDataset.CIFAR100 , format_dataset_params )
model = attack_pipeline.get_model(cifar_100_model_1, model_training_params)
attack_pipeline.run_attacks(model, attacks, AttackMethod.TF_PRIVACY , tf_attack_input_params)