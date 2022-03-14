import pandas as pd

import os

from datasets import load_dataset

# Important variables ------------------------------------------------------
#TODO: make variables into arguments passed from calling program

example_filepath = "./dataset/CrisisNLP_labeled_data_crowdflower/2013_Pakistan_eq/2013_Pakistan_eq_CF_labeled_data.tsv"
directory_of_original_datasets = "./dataset/CrisisNLP_labeled_data_crowdflower/"
dataset_complete_filepath = "./dataset/dataset_complete.csv"

# Functions to clean datasets
def clean_individual_dataset(filepath):
    df = pd.read_csv(filepath, sep="\t", header=0)
    df.index.name = "Index"

    labels = {
        'injured_or_dead_people' : 0,
        'missing_trapped_or_found_people' : 1,
        'displaced_people_and_evacuations' : 2,
        'infrastructure_and_utilities_damage' : 3,
        'donation_needs_or_offers_or_volunteering_services' : 4,
        'caution_and_advice' : 5,
        'sympathy_and_emotional_support' : 6,
        'other_useful_information': 7,
        'not_related_or_irrelevant' : 7,
    }


    df = df.drop(columns=['tweet_id'])
    # Convert labels to numbers (needed for model)
    df.replace(to_replace={"label": labels}, inplace=True) 
    return df


def obtain_filepaths(directory_of_datasets):
    filepaths = []
    for subdir, dirs, files in os.walk(directory_of_datasets):
        for file in files:
            if file.endswith(".tsv"):
                filepaths.append(os.path.join(subdir, file))
    return filepaths



def data_fusion(directory_of_dataset, dataset_complete_filepath):
    dataframes = []
    filepaths = obtain_filepaths(directory_of_dataset)
    for filepath in filepaths:
        dataframes.append(clean_individual_dataset(filepath))

    df_total = pd.concat(dataframes)

    # Rename columns 
    df_total.rename(columns={"label": "labels"}, inplace=True)
    df_total.rename(columns={"tweet_text": "text"}, inplace=True)
    
    df_total.to_csv(dataset_complete_filepath, index=False)


def run_create_csv(directory_of_original_datasets, dataset_complete_filepath):
    data_fusion(directory_of_original_datasets, dataset_complete_filepath)
    df = pd.read_csv(dataset_complete_filepath, header=0)
    #print(df.groupby('labels').count()) #Count of each label
    #print(df.groupby('labels').count()/18967) #percentage of each label




# Split dataset into train:dev:test
def split_dataset(dataset_complete_filepath):
    data = load_dataset("csv", data_files=dataset_complete_filepath)
    data = data["train"].train_test_split(train_size=0.8,
        seed=42, shuffle=True)

    data_test_valid = data['test'].train_test_split(train_size=0.5,
    shuffle=True)
    data['validation'] = data_test_valid.pop('train')
    data['test'] = data_test_valid.pop('test')
    return data


def save_dataset_apache_arrow(data):
    data.save_to_disk("./dataset/crisis_dataset/")




run_create_csv(directory_of_original_datasets, dataset_complete_filepath)

data = split_dataset(dataset_complete_filepath)
#print(data['train'].select(range(3))['text'])

save_dataset_apache_arrow(data)



