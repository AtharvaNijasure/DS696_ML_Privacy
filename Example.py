from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackType
from AttackPipeline import AttackPipeline
from DatasetRepo import RegisteredDataset
from Constants import *


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


attack_pipeline = AttackPipeline(cifar_100_model_1, RegisteredDataset.CIFAR100 , attacks,  format_dataset_params, model_training_params, tf_attack_input_params, ml_pr_attack_input_params )