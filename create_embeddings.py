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
        "--checkpoint", 
        type=str, 
        default="bert-base-cased", 
        help="Specify the checkpoint of your model e.g. bert-base-cased."
    )

    parser.add_argument(
        "--percent-dataset", 
        type=float, 
        default=1.0, 
        help="Percentage of the training data to use."
    )

    args = parser.parse_args()
    return args


#Helper function -------------------------------------------------------
def get_explanation_type(exp_dataset_filepath):
    if exp_dataset_filepath == "./dataset/crisis_dataset/exp/" or ("size" in
        exp_dataset_filepath):
        explanation_type = "normal"
    else:
        #e.g. ./dataset/crisis_dataset_few/exp/
        filename = exp_dataset_filepath.split("/")
        idx_explanation = [idx for idx, s in enumerate(filename) if 'crisis_dataset' in s][0]
        explanation_type = filename[idx_explanation].split("_")[-1]

    return explanation_type


# Main -------------------------------------------------------------------
def main():
    args = parse_args()

    explanation_type = get_explanation_type(args.exp_dataset_filepath)

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

    # Variables for ExpBERT embeddings --------------------------------------------

    num_datapoints = int(args.percent_dataset * 17117) #number of original datapoints of crisis dataset
    print(num_datapoints)
    dataset_size = raw_datasets.num_rows['train']

    if dataset_size == 616212:
        # number of explanations and textual descriptions
        num_exp_td = dataset_size / num_datapoints
    else:
        #TODO: Adapt so it works for any explanation set
        #for now, percent_dataset only works with 36 explanations
        num_exp_td = 616212 / 17117

    #Confirm it is a float that can be converted to int without rounding
    if num_exp_td % 1 != 0:
        raise ValueError("Need to provide the correct dataset size")
    
    num_exp_td = int(num_exp_td)



    # Embeddings NoExp ---------------------------------------------------

    if not os.path.exists('embeddings'):
        os.makedirs('embeddings')

    if args.checkpoint == "cardiffnlp/twitter-roberta-base":
        args.checkpoint = "twitter-roberta-base"


    

    if args.tiny_dataset:
        if args.exp_flag:
            dataset_size_sample = num_exp_td*5 
            num_datapoints = int(dataset_size_sample / num_exp_td)
            tokenized_train = \
                tokenizer(raw_datasets['train']['text'][:dataset_size_sample], truncation=True, padding=True, return_tensors='pt')
        else:
            tokenized_train = tokenizer(raw_datasets['train']['text'][:100], truncation=True, padding=True, return_tensors='pt')
    else:
        tokenized_train = tokenizer(raw_datasets['train']['text'], truncation=True, padding=True, return_tensors='pt')
    




    # NoExp ----------------------------------------------------------------

    if args.exp_flag is False:
        with torch.no_grad():
            train_ids = tokenized_train['input_ids']
            model_outputs = model(train_ids)
            
            #Embeddings is of dimensions number of tokens x 768 (output layer of BERT)
            output = model_outputs['last_hidden_state'] 
            
            #0 of last hidden layer is the CLS token
            embeddings = output[:, 0, :]
            
            torch.save(embeddings, f'./embeddings/noexp_{args.checkpoint}_embeddings.pt')
            exit(0)
   

    # ExpBERT embeddings ------------------------------------------------------

    with torch.no_grad():
        train_ids = tokenized_train['input_ids']
        
        #At this point we have tokenized all 671760 datapoints
        #We then pass them through the model and then restructure
        #the tensor to be (18660, num_explanations * 768)
        
        #Splits train_ids into tuple of Torch.Tensor
        
        #train_ids_split = torch.split(train_ids, int(train_ids.shape[0] / 100))
        #TODO: Try with num_ex_td length splits
        train_ids_split = torch.split(train_ids, num_exp_td)
    
        emb = [] 
        #Create embeddings by splitting train_ids
        for train_ids in train_ids_split:
            model_outputs = model(train_ids)
            #Embeddings is of dimensions number of tokens x 768 (output layer of BERT)
            output = model_outputs['last_hidden_state']
            #0 of last hidden layer is the CLS token
            embeddings = output[:,0,:]
            print(embeddings.shape)
            emb.append(embeddings)



        embeddings = torch.vstack(emb)
        print(embeddings.shape)

        """
        #model_outputs = model(train_ids)
 
        #Embeddings is of dimensions number of tokens x 768 (output layer of BERT)
        output = model_outputs['last_hidden_state']

        
        #0 of last hidden layer is the CLS token
        embeddings = output[:,0,:]
        """



        #embeddings = torch.reshape(embeddings, (num_datapoints, num_exp_td*768))

        #shape becomes num_datapoints x (num_explanations + num textual_descriptions) x 768
        embeddings = torch.reshape(embeddings, (num_datapoints, num_exp_td, 768))
        print(embeddings.shape)
        
        #Flatten tensors so that you have (num datapoints, num_exp_td x 768) 
        embeddings = torch.flatten(embeddings, start_dim=1)
        print(embeddings.shape)
        #Save embedding as pickle file 

        torch.save(embeddings,
        f'./embeddings/exp_{explanation_type}_{args.checkpoint}_embeddings.pt')


if __name__ == "__main__":
    main()
