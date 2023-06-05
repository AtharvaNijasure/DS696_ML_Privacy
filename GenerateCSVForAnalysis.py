import os
import pandas as pd

import csv

class RowObject:
    def __init__(self, dataset, data_slice, attack_method, training_samples, testing_samples, target_model_type,training_acc, training_f1, training_ps,	training_recall,	testing_acc,	testing_f1,	testing_ps,	testing_recall,	gen_gap  ):


        self.dataset = dataset
        self.data_slice = data_slice
        self.attack_method = attack_method
        self.training_samples = training_samples
        self.testing_samples = testing_samples
        self.target_model_type = target_model_type
        self.training_acc = training_acc
        self.training_f1 = training_f1
        self.training_ps = training_ps
        self.training_recall = training_recall
        self.testing_acc = testing_acc
        self.testing_f1 = testing_f1
        self.testing_ps = testing_ps
        self.testing_recall =testing_recall
        self.gen_gap = gen_gap
        self.AUC = 0.0
        self.PPV = 0.0
        self.ADV = 0.0



headers = ["Dataset","model","data_slice","attack_method","training_samples","testing_samples","metric_name","metric_value","target_model_type","training_acc","training_f1","training_ps","training_recall","testing_acc","testing_f1","testing_ps","testing_recall","gen_gap"]

# start reading csv
outfile  = "./Results/preFinalAdultRaw.csv"
infile = './Results/AdultIncomeTabularRaw.csv'
# Open the CSV file
# with open(infile, 'r') as file:
#     # Create a CSV reader object
#     reader = csv.reader(file)
#
#     # Iterate over each row in the CSV file
#     for row in reader:
#         # Print each value in the row
#         for value in row:
#             print(value, end=' ')
#         print()  # Print a new line after each row



data = list(csv.reader(open(infile)))

i = 1
final_list = []
final_combined_header = data[0]
final_combined_header.remove("attack_method")
final_combined_header.remove("metric_name")
final_combined_header.remove("metric_value")
final_list.append(final_combined_header)

'''
headers = ["Dataset","model","data_slice","attack_method","training_samples","testing_samples","metric_name","metric_value","target_model_type","training_acc","training_f1","training_ps","training_recall","testing_acc","testing_f1","testing_ps","testing_recall","gen_gap"]
            0           1       2           3                   4                   5               6                   7               8           9               10              11              12              13          14             15       16              17
'''
i=1
while( i < len(data)):

    z = 0

    metrics = {
        'THRESHOLD_ATTACK_AUC': '0',
        'THRESHOLD_ATTACK_PPV': '0',
        'THRESHOLD_ATTACK_advantage': '0'
    }
    dataslices = {

    }
    for rep in range(0,5) :
        final_res_row = []
        for z in range(0,3) :
            row = data[i]
            i+=1
            if z == 0 :
                final_res_row = row[:3].__add__(row[4:6]).__add__(row[8:])
            metric_name = row[6]
            attack_method = row[3]
            metric_value = row[7]
            metrics[attack_method+"_"+metric_name] = metric_value

        for key in metrics.keys() :
            final_res_row.append(metrics[key])
            dataslices[row[2]] = final_res_row


    final_list.append(final_res_row)


with open(outfile, "wb") as f:
    writer = csv.writer(f)
    writer.writerows(final_list)


