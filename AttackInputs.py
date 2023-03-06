from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackInputData
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import SlicingSpec
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackType

import tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.plotting as plotting

from DatasetRepo import DatasetRepo

'''
Note : try to use the pre-defined datastructures of tf_privacy and ml_privacy_meter as far as possible 

'''

class AttackInputs:


    def __init__(self):
        self.defined_attacks  = {



        }
        # self.data_set = DatasetRepo(dataset)




    def get_Attack_inputs(self,logits_train,logits_test , loss_train, loss_test,labels_train, labels_test  ) :
        input = AttackInputData(
                          logits_train = logits_train,
                          logits_test = logits_test,
                          loss_train = loss_train,
                          loss_test = loss_test,
                          labels_train = labels_train,
                          labels_test = labels_test
                        )




        return input

    def get_Attack_slicing_specs(self, entire_dataset, by_class, classification_correctness ):
        slicing_specs =  SlicingSpec(
                                    entire_dataset=entire_dataset,
                                    by_class=by_class,
                                    by_classification_correctness=classification_correctness
                                )

        return slicing_specs


    # def get_mi_attacks(self, attacks):
    #     attack_types = [AttackType.THRESHOLD_ATTACK,AttackType.LOGISTIC_REGRESSION]
    #     attack_types = []
    #     for attack in attacks :
    #
    #
    #
    #
    #     return attack_types