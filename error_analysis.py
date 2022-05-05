import numpy as np
import argparse
import random
from collections import Counter

from datasets import load_from_disk, load_metric, DatasetDict


def parse_args():
    parser = argparse.ArgumentParser(description="Run classifier")
    # Filepaths ------------------------------------------
    parser.add_argument(
        "--error-analysis-noexp-filepath", 
        type=str, 
        default="error_analysis/gcp/NoExp_WCEL/", 
        help="Location of NoExp error analysis numpy arrays."
    )
    
    parser.add_argument(
        "--error-analysis-exp-filepath", 
        type=str, 
        default="error_analysis/gcp/ExpBERT_WCEL/", 
        help="Location of ExpBERT error analysis numpy arrays."
    )

    parser.add_argument(
        "--noexp-dataset-filepath", 
        type=str, 
        default="./dataset/crisis_dataset/noexp/", 
        help="Location of Apache Arrow NoExp dataset."
    )

    args = parser.parse_args()
    return args


# Helper functions --------------------------------------

def print_lengths(cc, ic, ci, ii):
    print(len(cc))
    print(len(ic))
    print(len(ci))
    print(len(ii))
    #Total lengths should be 17117
    print(len(cc)+len(ic)+len(ci)+len(ii))


def print_counts(cc_labels, ic_labels, ci_labels, ii_labels):
    cc_counter = Counter(cc_labels)
    ic_counter = Counter(ic_labels)
    ci_counter = Counter(ci_labels)
    ii_counter = Counter(ii_labels)

    print(cc_counter)
    print(ic_counter)
    print(ci_counter)
    print(ii_counter)


def print_tweet(dpi, ds):
    #go to the original dataset and find a datapoint of interest
    print(f'TWEET of datapoint {dpi}')
    print(ds['train']['text'][dpi])
    print(f'Label of datapoint {dpi}')
    print(ds['train']['labels'][dpi])


def analyze_datapoints(classified_datapoints, sample_size, idx_noexp, raw_dataset):
    sample = random.sample(classified_datapoints, sample_size)
    for pair in sample:
        noexp = pair[0]
        #NoExp and ExpBERT have same tweet index
        tweet_index = idx_noexp[noexp]
        print_tweet(noexp, raw_dataset)



# Main --------------------------------------------------------------

def main():
    args = parse_args()
    preds_noexp = np.load(f'{args.error_analysis_noexp_filepath}/preds.npy')
    labels_noexp = np.load(f'{args.error_analysis_noexp_filepath}/labels.npy')
    idx_noexp = np.load(f'{args.error_analysis_noexp_filepath}/idx.npy')

    preds_exp = np.load(f'{args.error_analysis_exp_filepath}/preds.npy')
    labels_exp = np.load(f'{args.error_analysis_exp_filepath}/labels.npy')
    idx_exp = np.load(f'{args.error_analysis_exp_filepath}/idx.npy')


    #The index of the datapoint that we are interested in 
    #inspecting, before shuffling, in reality this point
    #is picked randomly, unless you

    #List of tuples (index of NoExp, index of ExpBERT)
    cc = [] #points were correct and remained correct
    ic = [] #points were incorrect and became correct
    ci = [] #points were correct and became incorrect
    ii = []
    
    cc_labels = []
    ic_labels = []
    ci_labels = []
    ii_labels = []

    #Find how many datapoints were correctly and 
    #incorrectly classified by NoExp
    for i in range(0, len(idx_noexp)):
        idx_noexp_shuffled = np.where(idx_noexp==i)[0][0]
        idx_exp_shuffled = np.where(idx_exp==i)[0][0]

        pred_noexp = preds_noexp[idx_noexp_shuffled]
        label_noexp = labels_noexp[idx_noexp_shuffled]

        pred_exp = preds_exp[idx_exp_shuffled]
        label_exp = labels_exp[idx_exp_shuffled]
        
        #If was correct and remained correct
        if (pred_noexp == label_noexp) and \
            (pred_exp == label_exp): 
            cc.append((idx_noexp_shuffled,idx_exp_shuffled))
            cc_labels.append(label_noexp)
        elif (pred_noexp != label_noexp) and \
            (pred_exp == label_exp):
            ic.append((idx_noexp_shuffled,idx_exp_shuffled))
            ic_labels.append(label_noexp)
        elif (pred_noexp == label_noexp) and \
            (pred_exp != label_exp):
            ci.append((idx_noexp_shuffled,idx_exp_shuffled))
            ci_labels.append(label_noexp)
        else:
            ii.append((idx_noexp_shuffled,idx_exp_shuffled))
            ii_labels.append(label_noexp)

    #Print counts of each class for each category
    print_counts(cc_labels, ic_labels, ci_labels, ii_labels)

    #Counts of each category
    #print_lengths(cc, ic, ci, ii)
    
    
    #Analyze random datapoints
    raw_dataset = load_from_disk(args.noexp_dataset_filepath)

    print("\nBoth correct") 
    analyze_datapoints(cc, 3, idx_noexp, raw_dataset)
    
    print("\nNoExp incorrect, ExpBERT correct")
    analyze_datapoints(ic, 3, idx_noexp, raw_dataset)

    print("\nNoExp correct, ExpBERT incorrect")
    analyze_datapoints(ci, 3, idx_noexp, raw_dataset)

    print("\nBoth incorrect")
    analyze_datapoints(ii, 3, idx_noexp, raw_dataset)




if __name__ == "__main__":
    main()
