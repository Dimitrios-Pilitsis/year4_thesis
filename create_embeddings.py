import os
from pathlib import Path
import argparse

from sklearn.metrics import ConfusionMatrixDisplay 

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from transformers import AutoTokenizer, AutoModel

from datasets import load_from_disk, load_metric, DatasetDict


# Argparser -----------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Create embeddings for" +\
    "classifiers")

    # Flags --------------------------------------------------------------
    parser.add_argument(
        '--exp-flag', 
        action='store_true', 
        help="Run ExpBERT"
    )

    parser.add_argument(
        "--tiny-dataset",
        action="store_true",
        help="Use smaller dataset for training and evaluation.",
    )


    # Directories -------------------------------------------------------------
    parser.add_argument(
        "--noexp-dataset-filepath", 
        type=str, 
        default="./dataset/crisis_dataset/noexp/", 
        help="Location of Apache Arrow NoExp dataset."
    )

    parser.add_argument(
        "--exp-dataset-filepath", 
        type=str, 
        default="./dataset/crisis_dataset/exp/", 
        help="Location of Apache Arrow Exp dataset."
    )

    # Other ---------------------------------------------------------------

    parser.add_argument(
        "--percent-dataset", 
        type=float, 
        default=1.0, 
        help="Percentage of the training data to use."
    )
   
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default="bert-base-cased", 
        help="Specify the checkpoint of your model e.g. bert-base-cased."
    )

    args = parser.parse_args()
    return args


# Main -------------------------------------------------------------------
def main():
    args = parse_args()

    #Model is set to evaluation mode by default using model.eval()
    #Using checkpoint is much quicker as model and tokenizer are cached by Huggingface
    model = AutoModel.from_pretrained(args.checkpoint,
        num_labels=9)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

    # Loading dataset ---------------------------------------------
    if args.exp_flag:
        raw_datasets = load_from_disk(args.exp_dataset_filepath)
    else:
        raw_datasets = load_from_disk(args.noexp_dataset_filepath)
    
    print(raw_datasets)

    # Embeddings NoExp ---------------------------------------------------

    if not os.path.exists('embeddings'):
        os.makedirs('embeddings')

    if args.exp_flag is False:
        if args.tiny_dataset:
            tokenized_train = tokenizer(raw_datasets['train']['text'][:10], truncation=True, padding=True, return_tensors='pt')
        else:
            tokenized_train = tokenizer(raw_datasets['train']['text'], truncation=True, padding=True, return_tensors='pt')

        with torch.no_grad():
            train_ids = tokenized_train['input_ids']
            model_outputs = model(train_ids)
            
            #Embeddings is of dimensions number of tokens x 768 (output layer of BERT)
            output = model_outputs['last_hidden_state'] 
            
            #0 of last hidden layer is the CLS token
            embeddings = output[:, 0, :]
            
            print(embeddings.shape)
            torch.save(embeddings, './embeddings/noexp_embeddings.pt')
            exit(0)
   

    # Variables for ExpBERT embeddings --------------------------------------------

    dataset_size = raw_datasets.num_rows['train']
    num_datapoints = 18660 #number of original daatapoints of crisis dataset
    # number of explanations and textual descriptions
    num_exp_td = dataset_size / num_datapoints 
    
    #Confirm it is a float that can be converted to int without rounding
    if num_exp_td % 1 != 0:
        raise ValueError("Need to provide the correct dataset size")
    
    num_exp_td = int(num_exp_td)


    # ExpBERT embeddings ------------------------------------------------------

    if args.tiny_dataset:
        dataset_size = num_exp_td*3 
        num_datapoints = int(dataset_size / num_exp_td)
        tokenized_train = tokenizer(raw_datasets['train']['text'][:dataset_size], truncation=True, padding=True, return_tensors='pt')
    else:
        tokenized_train = tokenizer(raw_datasets['train']['text'], truncation=True, padding=True, return_tensors='pt')
    

    with torch.no_grad():
        train_ids = tokenized_train['input_ids']
        model_outputs = model(train_ids)
        
        #Embeddings is of dimensions number of tokens x 768 (output layer of BERT)
        output = model_outputs['last_hidden_state']
        
        #0 of last hidden layer is the CLS token
        embeddings = output[:,0,:]
        
        #shape becomes num_datapoints x (num_explanations + num textual_descriptions) x 768
        embeddings = torch.reshape(embeddings, (num_datapoints, num_exp_td, 768))
        
        #Flatten tensors so that you have (num datapoints, num_exp_td x 768) 
        embeddings = torch.flatten(embeddings, start_dim=1)
        print(embeddings.shape)

        #Save embedding as pickle file 
        torch.save(embeddings, './embeddings/exp_embeddings.pt')


if __name__ == "__main__":
    main()
