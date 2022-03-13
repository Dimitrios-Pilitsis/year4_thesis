import pandas as pd

import os

example_filepath = "./dataset/CrisisNLP_labeled_data_crowdflower/2013_Pakistan_eq/2013_Pakistan_eq_CF_labeled_data.tsv"
directory_of_datasets = "./dataset/CrisisNLP_labeled_data_crowdflower/"





def clean_individual_dataset(filepath):
    df = pd.read_csv(filepath, sep="\t", header=0)
    df.index.name = "Index"

    labels = {
        'injured_or_dead_people' : 1,
        'missing_trapped_or_found_people' : 2,
        'displaced_people_and_evacuations' : 3,
        'infrastructure_and_utilities_damage' : 4,
        'donation_needs_or_offers_or_volunteering_services' : 5,
        'caution_and_advice' : 6,
        'sympathy_and_emotional_support' : 7,
        'other_useful_information': 8,
        'not_related_or_irrelevant' : 9,
    }


    df = df.drop(columns=['tweet_id'])

    # Rename label column (needed for training model)
    df.rename(columns={"label": "labels"}, inplace=True)

    # Convert labels to numbers (needed for model)
    df.replace(to_replace={"labels": labels}, inplace=True) 
    return df

def obtain_filepaths(directory_of_datasets):
    filepaths = []
    for subdir, dirs, files in os.walk(directory_of_datasets):
        for file in files:
            if file.endswith(".tsv"):
                filepaths.append(os.path.join(subdir, file))
    return filepaths



def data_fusion(directory_of_dataset):
    dataframes = []
    filepaths = obtain_filepaths(directory_of_dataset)
    for filepath in filepaths:
        dataframes.append(clean_individual_dataset(filepath))

    df_total = pd.concat(dataframes)
    df_total.to_csv('./dataset/dataset_complete.csv', index=False)



data_fusion(directory_of_datasets)


df1 = pd.read_csv("./dataset/dataset_complete.csv", header=0)
print(df1)
