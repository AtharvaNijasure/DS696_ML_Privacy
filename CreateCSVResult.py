import re
from Constants import *
import pandas as pd
import csv


class result_row:

    def __init__(self, model,data_slice ):

        self.model = model
        self.data_slice = data_slice
        self.attack_method = ""
        self.training_samples = ""
        self.testing_samples = ""
        self.metric_name = ""
        self.metric_value = ""




def getFileAsString(file_path) :
    with open(file_path, 'r') as f:
        text = f.read()
    return text

def extract_strings_from_file(file_path, start_flag, end_flag):
    with open(file_path, 'r') as f:
        text = f.read()
    pattern = re.compile(start_flag + '(.*?)' + end_flag)
    matches = pattern.findall(text)
    return matches

def get_model_name(text) :
    best_summary = "summary : Best-performing attacks over all slices"
    name = text.split(best_summary)[0]
    model_name = name.replace("\n", "")
    model_name = model_name.replace(model_placeholder, "")


    return model_name

def extract_results(row, rr):
    # "  THRESHOLD_ATTACK (with 27238 training and 6776 test examples) achieved an AUC of 0.50"
    # 0 1 2                3     4      5        6  7     8   9            10   11 12  13  14
    row = row.replace("positive predictive value" , "PPV")
    words = row.split(" ")
    rr.attack_method = words[2]
    rr.training_samples = words[4]
    rr.testing_samples = words[7]
    rr.metric_name = words[12]
    rr.metric_value = words[14]
    return


# script to create a csv file of results from output file

out_put_csv = "./Results/results_AdultDS.csv"

headers = []

start_delimiter = "------------***************"

end_delimiter = "***************------------\n"

model_placeholder = "New Summary : ./adult_income_models/"
summary_holder = "summary : Best-performing attacks over all slices"

best_perf = "Best-performing attacks over slice: "


# results = extract_strings_from_file(summary + ".txt", start_delimiter, end_delimiter)
read_file = "./Results/summaryAdultDS_tf_privacy_27_04"
text = getFileAsString( read_file + ".txt")

results = text.split(end_delimiter)

print(len(results))
# get tf privacy results
res_to_store = []
for res in results:
    if len(res) >10 :
        te = res.replace(start_delimiter, "")
        indiv_results = te.split(best_perf)
        mod_name = get_model_name(indiv_results[0])
        for i in range(1,len(indiv_results)) :
            # if len(ind) >2 :
            ind = indiv_results[i]
            rows = ind.split("\n")
            # there are 5 rows :
            data_slice = rows[0]
            for j in range(1, len(rows)) :
                row = rows[j]
                if(len(row) > 2) :
                    rr = result_row(mod_name , data_slice)
                    extract_results(row, rr)
                    res_to_store.append(rr)




# write results

with open(out_put_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["model", "data_slice", "attack_method", "training_samples", "testing_samples", "metric_name", "metric_value"])
    for rr in res_to_store:
        writer.writerow([rr.model, rr.data_slice, rr.attack_method, rr.training_samples, rr.testing_samples, rr.metric_name, rr.metric_value])



