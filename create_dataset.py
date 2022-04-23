import argparse

import pandas as pd
import numpy as np

import os

from datasets import load_dataset, load_from_disk
import demoji
import emoji

from visualizations import *

# argparser --------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Create dataset for NoExp or ExpBERT"+\
    "model on natural disaster tweet classification task")
 
    #Dataset sizing
    parser.add_argument(
        '--percent-dataset', 
        type=float, 
        default=1.0, 
        help="Specify percentage of original dataset desired for Exp dataset."
    )

    parser.add_argument(
        '--get-visualizations', 
        action="store_true",
        help="Specify whether you want to produce plots regarding the dataset."
    )

    # Directories and filepaths --------------------------------------------------------------
    parser.add_argument(
        '--explanations-filepath', 
        type=str, 
        default="explanations/explanations.txt", 
        help="Specify location of explanations text file"
    )
    
    parser.add_argument(
        '--original-dataset-filepath', 
        type=str, 
        default="./dataset/CrisisNLP_labeled_data_crowdflower/", 
        help="Specify location of original dataset"
    )

    parser.add_argument(
        '--noexp-csv-filepath', 
        type=str, 
        default="./dataset/dataset_noexp.csv", 
        help="Specify location of NoExp csv file"
    )

    parser.add_argument(
        '--exp-csv-filepath', 
        type=str, 
        default="./dataset/dataset_exp.csv", 
        help="Specify location of Exp csv file"
    )

    parser.add_argument(
        '--output-noexp-directory', 
        type=str, 
        default="./dataset/crisis_dataset/noexp/", 
        help="Specify location of NoExp Apache Arrow directory"
    )

    parser.add_argument(
        '--output-exp-directory', 
        type=str, 
        default="./dataset/crisis_dataset/exp/", 
        help="Specify location of Exp Apache Arrow directory"
    )

    args = parser.parse_args()

    return args


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
        'other useful information',
        'not related or irrelevant',
    ]
    
    ex_td = explanations + textual_descriptions
    len_df = len(df.index)
    #Create ex_td times each row
    df = df.iloc[np.repeat(np.arange(len(df)), len(ex_td))].reset_index(drop=True) 

    ex_td = ex_td * len_df
    #Add each explanation and textual description to each datapoint
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
def data_fusion(args):
    dataframes = []
    filepaths = obtain_filepaths(args.original_dataset_filepath)
    for filepath in filepaths:
        df = clean_individual_dataset(filepath)
        dataframes.append(df)

    df_total = pd.concat(dataframes)

    # Rename columns 
    df_total.rename(columns={"label": "labels"}, inplace=True)
    df_total.rename(columns={"tweet_text": "text"}, inplace=True)

    #Check for duplicate tweets
    df_noexp = check_for_duplicate_tweets(df_total)
    df_noexp.drop_duplicates(subset=['text'], inplace=True)
    #df_total.drop_duplicates(subset=['text'], inplace=True)
    
    #Get distributions and counts of labels
    #inspect_labels(df_total)

    #Sample dataset if provided with percent dataset argument
    if args.percent_dataset != 1.0:
        df_noexp = df_noexp.sample(frac=1).reset_index(drop=True) #Shuffle in place
        df_noexp = df_noexp.head(int(len(df_noexp)*(args.percent_dataset))) #Get first percent of dataframe
    
    explanations = read_explanations(args.explanations_filepath)
    df_exp = create_explanations_dataset(df_noexp, explanations)
    df_noexp.to_csv(args.noexp_csv_filepath, index=False)
    df_exp.to_csv(args.exp_csv_filepath, index=False)


# Main -------------------------------------------------------------
def main():
    args = parse_args()

    if not os.path.exists('plots'):
        os.makedirs('plots')

    data_fusion(args)

    data_noexp = load_dataset("csv", data_files=args.noexp_csv_filepath)
    data_exp = load_dataset("csv", data_files=args.exp_csv_filepath)
    data_noexp.save_to_disk(args.output_noexp_directory)
    data_exp.save_to_disk(args.output_exp_directory)
    
    visualizations_dataset(args.noexp_csv_filepath,
        args.exp_csv_filepath)

    label_distribution_pie_chart(args.noexp_csv_filepath)

if __name__ == "__main__":
    main()
