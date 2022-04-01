import pandas as pd

import numpy as np

import os

from datasets import load_dataset

import demoji
import emoji

from visualizations import *

# Important variables ------------------------------------------------------
#TODO: make variables into arguments passed from calling program

example_filepath = "./dataset/CrisisNLP_labeled_data_crowdflower/2013_Pakistan_eq/2013_Pakistan_eq_CF_labeled_data.tsv"
directory_of_original_datasets = "./dataset/CrisisNLP_labeled_data_crowdflower/"
dataset_complete_noexp_filepath = "./dataset/dataset_complete_noexp.csv"
dataset_complete_exp_filepath = "./dataset/dataset_complete_exp.csv"

explanations_filepath = "./explanations.txt"



# Helper functions for inspecting data -------------------------------
def see_datapoints(filepath, label):
    pd.set_option('display.max_colwidth', None)
    df = pd.read_csv(filepath, header=0)
    print(df)
    deaths = df.loc[df['labels'] == label]['text']
    print(deaths)



def inspect_labels(df):
    print(df.groupby('labels').count()) #count of each label
    print(df.groupby('labels').count()/len(df.index)) #percentage of each label



# Functions to clean tweets - -----------------------------------------------
def camel_case_split(word):
    start_idx = [i for i, e in enumerate(word) if e.isupper()] + [len(word)]
    start_idx = [0] + start_idx
    list_words = [word[x: y] for x, y in zip(start_idx, start_idx[1:])][1:] 
    return ' '.join(list_words)



#Deals with cases where there are consecutive emojis or no space between text
#and emoji
def emoji_present(text):
    if emoji.is_emoji(text) or (len(text) > 1 and emoji.is_emoji(text[0])) or (len(text) > 1 and emoji.is_emoji(text[-1])):
        return demoji.replace_with_desc(text, sep="")
    else:
        return text



def placeholders(texts):
    for count, text in enumerate(texts):
        new_text = []
        for t in text.split(" "):
            t = '' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            t = "" if "RT" in t else t
            t = camel_case_split(t) if t.startswith('#') and len(t) > 1 else t
            t = emoji_present(t)
            new_text.append(t)

        texts[count] = " ".join(new_text).strip()

    return texts


# Clean datasets -----------------------------------------------

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
        'not_related_or_irrelevant' : 8,
    }

    df = df.drop(columns=['tweet_id'])
    # Convert labels to numbers (needed for model)
    df.replace(to_replace={"label": labels}, inplace=True) 

    df = df.astype({'tweet_text': 'string'})

    # Add placeholders and remove unnecessary text
    df['tweet_text'] = placeholders(df['tweet_text'])

    #Remove empty tweets
    df['tweet_text'].replace('', np.nan, inplace=True) 
    df.dropna(subset=['tweet_text'], inplace=True)
    
    return df



# Explanation functions --------------------------------

def read_explanations(explanation_file):
    with open(explanation_file, 'r') as reader:
        lines = reader.readlines()
        explanations = [line.strip() for line in lines]
        return explanations

def create_explanations_dataset(df, explanations):
    textual_descriptions = [
        'injured or dead people',
        'missing trapped or found people',
        'displaced people an evacuations',
        'infrastructure and utilities damage',
        'donation needs or offers or volunteering services',
        'caution and advice', 
        'sympathy and emotional support', 
        'other_useful_information',
        'not related or irrelevant',
    ]
    ex_td = textual_descriptions + explanations
    len_df = len(df.index)
    df = pd.concat([df]*len(ex_td), ignore_index=True)
    ex_td = ex_td * len_df
    df.insert(1, "exp_and_td", ex_td, allow_duplicates = True)
    return df

# Dataset helper functions ----------------------------------

def obtain_filepaths(directory_of_datasets):
    filepaths = []
    for subdir, dirs, files in os.walk(directory_of_datasets):
        for file in files:
            if file.endswith(".tsv"):
                filepaths.append(os.path.join(subdir, file))
    return filepaths


def check_for_duplicate_tweets(df):
    df.astype({'text': 'string'})
    df.drop_duplicates(subset=['text'], inplace=True)
    #Checks for duplicates
    #duplicates = df[df.duplicated(keep="first", subset=['text'])]
    return df


# Clean dataset functions -------------------------------------
def data_fusion(directory_of_dataset, explanations_filepath, ds_noexp_fp,
    ds_exp_fp):
    dataframes = []
    filepaths = obtain_filepaths(directory_of_dataset)
    for filepath in filepaths:
        df = clean_individual_dataset(filepath)
        dataframes.append(df)

    df_total = pd.concat(dataframes)

    # Rename columns 
    df_total.rename(columns={"label": "labels"}, inplace=True)
    df_total.rename(columns={"tweet_text": "text"}, inplace=True)

    #Check for duplicate tweets
    df_noexp = check_for_duplicate_tweets(df_total)
    
    #Get distributions and counts of labels
    #inspect_labels(df_total)

    explanations = read_explanations(explanations_filepath)
    df_exp = create_explanations_dataset(df_total, explanations)

    df_noexp.to_csv(ds_noexp_fp, index=False)
    df_exp.to_csv(ds_exp_fp, index=False)


def run_create_csv(directory_of_original_datasets, explanations_filepath, ds_noexp_fp,
    ds_exp_fp):
    data_fusion(directory_of_original_datasets, explanations_filepath,
        ds_noexp_fp, ds_exp_fp)
    

# Split dataset into train:test
def split_dataset(dataset_complete_filepath):
    data = load_dataset("csv", data_files=dataset_complete_filepath)
    data = data["train"].train_test_split(train_size=0.8,
        seed=42, shuffle=True)
    return data


# Main -------------------------------------------------------------
def main():
    #visualizations_dataset(dataset_complete_noexp_filepath,
    #    dataset_complete_exp_filepath)

    if not os.path.exists('plots'):
        os.makedirs('plots')

    run_create_csv(directory_of_original_datasets, explanations_filepath,
        dataset_complete_noexp_filepath, dataset_complete_exp_filepath)


    data_noexp = split_dataset(dataset_complete_noexp_filepath)
    data_exp = split_dataset(dataset_complete_exp_filepath)

    # Check random points
    #print(data['train'].select(range(3))['text'])

    data_noexp.save_to_disk("./dataset/crisis_dataset/noexp/")
    data_exp.save_to_disk("./dataset/crisis_dataset/exp/")
    
    visualizations_dataset(dataset_complete_noexp_filepath,
        dataset_complete_exp_filepath)

if __name__ == "__main__":
    main()
