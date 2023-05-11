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





"""
attack pipeline - init(dataset,attack,model_train_and_dataset_params = {})
get_model(model_fx, training_params)
run_other_models(#only for reference and shadow)
run_attack(target_model,attack_params)
"""



epoch_list = [30,40,50,60] # 10,20,
batch_sizes =  [ 64 ] # 256
num_neighs = [3,4,5,10,15]
hidden_layers_l = [(32),(64), (64,128), (64,128, 256), (64,128,32)]

model_names = [ "sk_learn_KNN", "sk_learn_MLP", "decisionTreeCLS", "sk_learn_LR" ] #
# "sk_learn_KNN", "sk_learn_MLP", "decisionTreeCLS", "sk_learn_LR",
# "model_basic_LR_1_titanic",model_basic_MLP_1 ,
dict =  {
            "sk_learn_KNN" : [3,4,5,10,15],
            "sk_learn_MLP" :[(32),(64), (64,128), (64,128, 256), (64,128,32)],
            "decisionTreeCLS" : [3,4,5,10,15],
            "sk_learn_LR" : [1]
}



# wrapper
# model = attack_pipeline.get_model(cifar_100_model_1, model_training_params)
# attack_pipeline.run_attacks(model, attacks_tf_p,model_training_params, AttackMethod.TF_PRIVACY , tf_attack_input_params)
# attack_pipeline.run_attacks(model, attacks_ml_pr,model_training_params, AttackMethod.ML_PRIVACY , ml_pr_attack_input_params)

for mod_name in model_names :
    print(f"Started {mod_name}")
    # for ep in epoch_list :
    for bs in batch_sizes :
        l = 0
        list  = dict[mod_name]
        for nl in list :
            print(f"Started {mod_name} for   batch_size = {bs}  l = {l}") # epoch :{ep} max_depth :{nl}

            attack_parameters_titanic = {
                batch_size: bs,
                num_population_points: 400,
                fpr_tolerance_list: [
                    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
                ],
                loss_fn: tf.keras.losses.CategoricalCrossentropy(),
                model_type: ModelType.SkLearn
            }
            dataset_parameters = {
                num_train_points: 200,
                num_test_points: 200,
                verbose: 2,
                "train_size": 0.8
            }

            model_training_params = {

                verbose: 1,
                model_hyper_param: True

            }

            if(mod_name == "sk_learn_KNN") :
                model_training_params[num_neigh] = nl
            if (mod_name == "sk_learn_MLP"):
                model_training_params[hidden_layers] = nl
            if (mod_name == "decisionTreeCLS"):
                model_training_params[max_depth] = nl
            if (mod_name == "sk_learn_LR"):
                model_training_params[model_hyper_param] = False

            for att in attacks_tf_p :
                print(f"Started attack {att} for {mod_name}  l = {l}") # epoch :{ep}  max_depth :{nl}
                attack_pipeline_population = AttackPipeline(
                    RegisteredDataset.WINEQUALITY, att, dataset_parameters
                )
                model = attack_pipeline_population.get_model(mod_name, model_training_params)
                # break
                model_file = attack_pipeline_population.get_model_file(mod_name, model_training_params)
                attack_pipeline_population.run_attack(model, attack_parameters_titanic, model_file_name = model_file)
            l+=1
    print(f"Done {mod_name}")