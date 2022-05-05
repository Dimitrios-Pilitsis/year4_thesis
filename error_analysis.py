import numpy as np
import argparse

from datasets import load_from_disk, load_metric, DatasetDict




def parse_args():
    parser = argparse.ArgumentParser(description="Run classifier")

    parser.add_argument(
        "--datapoint-of-interest", 
        type=int, 
        default=0, 
        help="The datapoint index you want to find out how NoExp and ExpBERT predicted."
    )

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



def print_datapoint(dpi, ds):
    #go to the original dataset and find a datapoint of interest
    print(f'TWEET of datapoint {dpi}')
    print(ds['train']['text'][dpi])
    print(f'Label of datapoint {dpi}')
    print(ds['train']['labels'][dpi])


def main():
    args = parse_args()
    preds_noexp = np.load(f'{args.error_analysis_noexp_filepath}/preds.npy')
    labels_noexp = np.load(f'{args.error_analysis_noexp_filepath}/labels.npy')
    idx_noexp = np.load(f'{args.error_analysis_noexp_filepath}/idx.npy')

    preds_exp = np.load(f'{args.error_analysis_exp_filepath}/preds.npy')
    labels_exp = np.load(f'{args.error_analysis_exp_filepath}/labels.npy')
    idx_exp = np.load(f'{args.error_analysis_exp_filepath}/idx.npy')



    #See how many points model got correct
    print(np.sum(preds_noexp == labels_noexp))
    
    #See how many points model got correct
    print(np.sum(preds_exp == labels_exp))


    #TODO: Get actual datapoint using huggingface


    raw_dataset = load_from_disk(args.noexp_dataset_filepath)

    #The index of the datapoint,
    #that we are interested in inspecting, before shuffling
    #in reality this point is picked randomly, unless you
    
    print_datapoint(args.datapoint_of_interest, raw_dataset)
    exit(0)



    #TODO: Check if ExpBERT improved a datapoint

    
    idx_datapoint = np.where(idx==args.datapoint_of_interest)
    print(idx_datapoint)

    #Issue is that we don't know if its in train or test set, need to note it
    #print(train_dataset[idx_datapoint])



if __name__ == "__main__":
    main()
