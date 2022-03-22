import pandas as pd

import numpy as np

import os

from datasets import load_dataset

import demoji
import emoji

# Important variables ------------------------------------------------------
#TODO: make variables into arguments passed from calling program

example_filepath = "./dataset/CrisisNLP_labeled_data_crowdflower/2013_Pakistan_eq/2013_Pakistan_eq_CF_labeled_data.tsv"
directory_of_original_datasets = "./dataset/CrisisNLP_labeled_data_crowdflower/"
dataset_complete_filepath = "./dataset/dataset_complete.csv"

# Functions to clean datasets


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

    df = df.astype({'tweet_text': 'string'})

    # Add placeholders and remove unnecessary text
    df['tweet_text'] = placeholders(df['tweet_text'])

    #Remove empty tweets
    df['tweet_text'].replace('', np.nan, inplace=True) 
    df.dropna(subset=['tweet_text'], inplace=True)

    
    return df


def check_individual_dataset():
    file_path = "/home/dimipili/Documents/Documents/my_folder/University/Year_4/Thesis/year4_thesis/dataset/CrisisNLP_labeled_data_crowdflower/2013_Pakistan_eq/2013_Pakistan_eq_CF_labeled_data.tsv"
    df = clean_individual_dataset(filepath)
    print(df)
    print((df['tweet_text'][1361]))





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


def data_fusion(directory_of_dataset, dataset_complete_filepath):
    dataframes = []
    filepaths = obtain_filepaths(directory_of_dataset)
    for filepath in filepaths:
        dataframes.append(clean_individual_dataset(filepath))

    df_total = pd.concat(dataframes)

    # Rename columns 
    df_total.rename(columns={"label": "labels"}, inplace=True)
    df_total.rename(columns={"tweet_text": "text"}, inplace=True)

    #Check for duplicate tweets
    df_total = check_for_duplicate_tweets(df_total)
    
    df_total.to_csv(dataset_complete_filepath, index=False)


def run_create_csv(directory_of_original_datasets, dataset_complete_filepath):
    data_fusion(directory_of_original_datasets, dataset_complete_filepath)
    df = pd.read_csv(dataset_complete_filepath, header=0)
    #print(df.groupby('labels').count()) #Count of each label
    #print(df.groupby('labels').count()/18967) #percentage of each label




# Split dataset into train:eval
def split_dataset(dataset_complete_filepath):
    data = load_dataset("csv", data_files=dataset_complete_filepath)
    data = data["train"].train_test_split(train_size=0.8,
        seed=42, shuffle=True)
    return data


def save_dataset_apache_arrow(data):
    data.save_to_disk("./dataset/crisis_dataset/")




run_create_csv(directory_of_original_datasets, dataset_complete_filepath)

data = split_dataset(dataset_complete_filepath)

# Check random points
#print(data['train'].select(range(3))['text'])

save_dataset_apache_arrow(data)



