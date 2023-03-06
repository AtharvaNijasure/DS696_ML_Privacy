from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import membership_inference_attack as mia
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackInputData
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import SlicingSpec
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackType

import tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.plotting as plotting

class TfWrapper:

    def __init__(self, attacks, ):

        # self.slicing_specs =

        self.attack_types = attacks









# # Run several attacks for different data slices
# attacks_result = mia.run_attacks(input,
#                                  SlicingSpec(
#                                      entire_dataset = True,
#                                      by_class = True,
#                                      by_classification_correctness = True
#                                  ),
#                                  attack_types = [
#                                      AttackType.THRESHOLD_ATTACK,
#                                      AttackType.LOGISTIC_REGRESSION])
#
# # Plot the ROC curve of the best classifier
# fig = plotting.plot_roc_curve(
#     attacks_result.get_result_with_max_auc().roc_curve)
#
# # Print a user-friendly summary of the attacks
# print(attacks_result.summary(by_slices = True))


