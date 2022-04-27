import os
import argparse
import time

import numpy as np

import torch
from transformers import AutoTokenizer, AutoModel

from datasets import load_from_disk, DatasetDict




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
        help="Use tiny-dataset to confirm program works correctly.",
    )
    
    parser.add_argument(
        '--timer', 
        action="store_true",
        help="Specify whether you want to see the runtime of the program."
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

    parser.add_argument(
        "--split-value", 
        type=float, 
        default=150, 
        help="How much to split train_ids before obtaining embeddings."
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

# Sorting -----------------------------------------------

def filepath_keys(text):
    val = int(text.split("/")[-1].split(".")[0])
    return val

#filepath_keys("./embeddings/exp_normal_bert-base-cased/142.pt")
#exit(0)


# Tokenizers --------------------------------------------
def decode_text(tokenizer, text):
    encoded_input = tokenizer(text)
    decoded_text = tokenizer.decode(encoded_input["input_ids"])
    return decoded_text


def create_tiny_dataset(raw_datasets, args, num_exp_td):
    if args.exp_flag:
        dataset_size_sample = num_exp_td*5 
        num_datapoints = int(dataset_size_sample / num_exp_td)
        raw_datasets = raw_datasets.shuffle()['train'][:dataset_size_sample]
    else:
        raw_datasets = raw_datasets.shuffle()['train'][:200]

    raw_datasets = DatasetDict({'train' : raw_datasets})

    return raw_datasets


# Main -------------------------------------------------------------------
def main():
    start_time = time.time()
    args = parse_args()
    
    explanation_type = get_explanation_type(args.exp_dataset_filepath)

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
    dataset_size = raw_datasets.num_rows['train']

    num_exp_td = int(dataset_size / num_datapoints)


    # Create tokenized dataset ---------------------------------------------------

    if not os.path.exists('embeddings'):
        os.makedirs('embeddings')

    if args.checkpoint == "cardiffnlp/twitter-roberta-base":
        args.checkpoint = "twitter-roberta-base"
    
    def tokenize_noexp_function(examples):
            return tokenizer(examples["text"], truncation=True, padding=True,
                return_tensors='pt')

    def tokenize_exp_function(examples):
        return tokenizer(examples['text'], examples['exp_and_td'],
            truncation=True, padding=True, return_tensors='pt')

    if args.tiny_dataset:
        raw_datasets = create_tiny_dataset(raw_datasets, args, num_exp_td)

    if args.exp_flag:
        tokenized_train = \
            tokenize_exp_function(raw_datasets['train'])
    else:
        tokenized_train = \
            tokenize_noexp_function(raw_datasets['train'])

    torch.backends.cudnn.benchmark = True
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    torch.cuda.empty_cache()

    # Create embeddings ------------------------------------------------------

    with torch.no_grad():
        train_ids = tokenized_train['input_ids']
        train_ids = train_ids.to(device)
        
        #Splits train_ids into tuple of Torch.Tensor
        train_ids_split = torch.split(train_ids, int(train_ids.shape[0] / args.split_value))
        
        emb = [] 
        #Create embeddings through smaller train_ids
        for count, train_ids in enumerate(train_ids_split):
            model_outputs = model(train_ids)
            #Embeddings is of dimensions number of tokens x 768 (output layer of BERT)
            output = model_outputs['last_hidden_state']
            #0 of last hidden layer is the CLS token
            embeddings = output[:,0,:]
            print(count, embeddings.shape)
            #Numpy arrays use significantly less space than Tensors
            embeddings = embeddings.cpu().detach().numpy()
            emb.append(embeddings)
            torch.cuda.empty_cache()
        
        #Stack into (num datapoints, 768)
        emb = np.array(emb)
        emb = np.vstack(emb)
        
        embeddings = torch.tensor(emb)
        print(embeddings.shape)
        
        #NoExp ends here
        if args.exp_flag is False:
            torch.save(embeddings, f'./embeddings/noexp_{args.checkpoint}_embeddings.pt')

            if args.timer:
                duration = time.time() - start_time
                print(f'Program took {duration} seconds to run')

            exit(0)

        #Reshape to expect for instance (17117,36*768) i.e. have 1 unique tweet
        #Per row of tensor
        embeddings = torch.reshape(embeddings, (num_datapoints, num_exp_td*768))
        print(embeddings.shape)

        #Save final embedding as pickle file 
        embeddings_filepath = f'./embeddings/exp_{explanation_type}_{args.checkpoint}'
        torch.save(embeddings,
            f'{embeddings_filepath}_embeddings.pt')
    
        if args.timer:
            duration = time.time() - start_time
            print(f'Program took {duration} seconds to run')





if __name__ == "__main__":
    main()
