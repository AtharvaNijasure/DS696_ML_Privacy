from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackType
import AttackPipeline



attacks = [AttackType.THRESHOLD_ATTACK,
                                     AttackType.LOGISTIC_REGRESSION]



attack_pipeline = AttackPipeline("cifar100_model_1", "cifar100" , attacks )