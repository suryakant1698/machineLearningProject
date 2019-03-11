from time import time, sleep
from argparse import ArgumentParser
from os import listdir
from classifier import classifier


def get_input_args():
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, default='pet_images/', help='path to the folder of the images')
    parser.add_argument('--cnn', type=str, default='alexnet', help='the cnn model architecture to use')
    parser.add_argument('--file', type=str, default='dognames.txt')
    args = parser.parse_args()
    return args


def get_pet_labels(direc):
    results_dic = dict()
    filename_list = listdir(direc)
    for filename in filename_list:
        lis_pet_image = filename.lower().split('_')
        pet_name = ""
        for temp in lis_pet_image:
            if temp.isalpha():
                pet_name += temp + " "
        pet_name = pet_name.strip()
        if filename not in results_dic:
            results_dic[filename] = [pet_name]
    return results_dic


def classify_images(direc, rsults, arch):
    for key in results:
        pet_label = results[key][0].lower()
        classifier_label = classifier(direc+key, arch).lower()
        results[key].append(classifier_label)
        if pet_label in classifier_label:
            results[key].append(1)
        else:
            results[key].append(0)


def isadog(file, results):
    dognames_dic = dict()
    with open(file, "r") as inFile:
        line = inFile.readline()
        while line!="":
            line = line.strip()
            if line not in dognames_dic:
                dognames_dic[line] = 1
            line = inFile.readline()
    for key in results:
        if results[key][0] in dognames_dic:
            if results[key][1] in dognames_dic:
                results[key].extend((1,1))     
            else: 
                results[key].extend((1,0))
        else:
            if results[key][1] in dognames_dic:
                results[key].extend((0,1))
            else:
                results[key].extend((0,0))


def calculate_results(results):
    results_stats_dic = dict()
    results_stats_dic['n_dogs_img'] = 0
    results_stats_dic['n_match'] = 0
    results_stats_dic['n_correct_dogs'] = 0
    results_stats_dic['n_correct_notdogs'] = 0
    results_stats_dic['n_correct_breed'] = 0  
    for key in results:
        if results[key][2] == 1:
            results_stats_dic['n_match']+=1
            if results[key][3] == 1:
                results_stats_dic['n_correct_breed']+=1
        if results[key][3] == 1:
            results_stats_dic['n_dogs_img']+=1
            if results[key][4] == 1:
                results_stats_dic['n_correct_dogs']+=1
        else:
            results_stats_dic['n_correct_notdogs']+=1
    results_stats_dic['n_images'] = len(results)
    results_stats_dic['n_notdogs_img'] = (results_stats_dic['n_images']-results_stats_dic['n_dogs_img'])
    results_stats_dic['pct_match'] = results_stats_dic['n_match']/results_stats_dic['n_images']*100
    results_stats_dic['pct_correct_dogs'] = results_stats_dic['n_correct_dogs']/results_stats_dic['n_dogs_img']*100
    results_stats_dic['pct_correct_breed'] = results_stats_dic['n_correct_breed']/results_stats_dic['n_dogs_img']*100
    if results_stats_dic['n_notdogs_img'] > 0:
        results_stats_dic['pct_correct_notdogs'] = results_stats_dic['n_correct_notdogs']/results_stats_dic['n_notdogs_img']*100
    else:
        results_stats_dic['pct_correct_notdogs'] = 0.0
    return results_stats_dic

def print_result(results, results_stats_dic, model, print_incorrect_dogs = False, print_incorrect_breed = False):
    print("\n\n*** Results Summary for CNN Model Architecture", model.upper(), "***")
    print("{:20}: {:3d}".format('N Images', results_stats_dic['n_images']))
    print("{:20}: {:3d}".format('N Dog Images', results_stats_dic['n_dogs_img']))
    print("N Not dog images:",results_stats_dic['n_notdogs_img'])
    for key in results_stats_dic:
        if key[0] == 'p':
            print(key,results_stats_dic[key])
    if print_incorrect_dogs == True and (results_stats_dic['n_correct_dogs'] + results_stats_dic['n_correct_notdogs'] != 
        results_stats_dic['n_images']):
        print("\nINCORRECT Dog/NOT Dog Assignments:")
    for key in  results:
        if results[key][3] != results[key][4]:
            print(results[key][0],results[key][1])
    if (print_incorrect_breed and 
    (results_stats_dic['n_correct_dogs'] != results_stats_dic['n_correct_breed']) 
    ):
        print("\nINCORRECT Dog Breed Assignment:") 
    for key in results:
            # Pet Image Label is-a-Dog, classified as-a-dog but is WRONG breed
        if ( sum(results[key][3:]) == 2 and results[key][2] == 0 ):                
            print("Real: {:>26}   Classifier: {:>30}".format(results[key][0], results[key][1]))

start_time = time()
in_args = get_input_args()
results = get_pet_labels(in_args.dir)
classify_images(in_args.dir, results, in_args.cnn)
isadog(in_args.file, results)
stats = calculate_results(results)
print_result(results, stats, in_args.cnn)
#print(stats)
#print(results)
end_time = time()