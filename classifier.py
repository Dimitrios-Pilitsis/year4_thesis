import argparse

import torch

import numpy as np

from torch import nn
from torch.nn import functional as F
from typing import Callable
from torch import optim

from datasets import load_from_disk

# Argparser --------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Run classifier")

    parser.add_argument(
        "--noexp-embeddings-filepath", 
        type=str, 
        default="embeddings/noexp_embeddings.pt", 
        help="Location of Apache Arrow NoExp dataset."
    )

    parser.add_argument(
        "--exp-embeddings-filepath", 
        type=str, 
        default="./embeddings/exp_embeddings.pt", 
        help="Location of Apache Arrow Exp dataset."
    )
    
    parser.add_argument(
        "--noexp-dataset-filepath", 
        type=str, 
        default="./dataset/crisis_dataset/noexp/", 
        help="Location of Apache Arrow No Exp dataset."
    )  

    parser.add_argument(
        "--exp-dataset-filepath", 
        type=str, 
        default="./dataset/crisis_dataset/exp/", 
        help="Location of Apache Arrow Exp dataset."
    )

    parser.add_argument(
        '--exp-flag', 
        action='store_true', 
        help="Run ExpBERT"
    )

    parser.add_argument(
        "--percent-dataset", 
        type=float, 
        default=1.0, 
        help="Percentage of the training data to use."
    )


    args = parser.parse_args()
    return args


# Shuffle and sample ----------------------------------------
#Shuffle data
def shuffle_data(embeddings, labels, percent_dataset):
    vals = torch.arange(0, embeddings.shape[0], dtype=float) #Tensor of
    #indices in order to shuffle and get randomized points from embeddings
    idx = torch.multinomial(vals, num_samples=vals.shape[0], replacement=False)
    embeddings = embeddings[idx]

    #Shuffle labels
    labels = np.array(labels).reshape((-1, 1))
    idx_list = idx.tolist()
    labels = labels[idx_list]

    if percent_dataset != 1.0:
        sample = int(embeddings.shape[0] * percent_dataset)
        embeddings = embeddings[:sample,:]

    return embeddings, labels


# Arrange tensor into train and test -------------------------------

def train_test_split(embeddings, labels):
    train_split = int(0.7*embeddings.shape[0])
    print(train_split)

    train_embeddings = embeddings[:train_split].type(torch.float32)
    test_embeddings = embeddings[train_split:].type(torch.float32)

    train_labels = torch.tensor(labels[:train_split], dtype=torch.long)
    test_labels = torch.tensor(labels[train_split:], dtype=torch.long)


    features = {
        'train': train_embeddings,
        'test' : test_embeddings,
    }
    labels = {
        'train': train_labels,
        'test': test_labels,
    }
    return features, labels


# Preprocess overall -------------------------------------
def preprocess(exp_flag, exp_embeddings_filepath, noexp_embeddings_filepath,
    exp_dataset_filepath, noexp_dataset_filepath, percent_dataset):

    if exp_flag:
        embeddings = torch.load(exp_embeddings_filepath)
        raw_datasets = load_from_disk(exp_dataset_filepath)
    else:
        embeddings = torch.load(noexp_embeddings_filepath)
        raw_datasets = load_from_disk(noexp_dataset_filepath)

    labels = raw_datasets['train']['labels']

    embeddings, labels = shuffle_data(embeddings, labels, percent_dataset)
    features, labels = train_test_split(embeddings, labels)
    return features, labels

# Classifier --------------------------------------------


#torch.backends.cudnn.benchmark = True



# Main --------------------------------------------------

def main():
    args = parse_args()
    
    features, labels = preprocess(args.exp_flag,
        args.exp_embeddings_filepath, args.noexp_embeddings_filepath,
        args.exp_dataset_filepath, args.noexp_dataset_filepath,
        args.percent_dataset)

    print(features)






if __name__ == "__main__":
    main()
