from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackType
from AttackPipeline import AttackPipeline
from DatasetRepo import RegisteredDataset



attacks = [AttackType.THRESHOLD_ATTACK,
                                     AttackType.LOGISTIC_REGRESSION]



attack_pipeline = AttackPipeline("cifar100_model_1", RegisteredDataset.CIFAR100 , attacks )