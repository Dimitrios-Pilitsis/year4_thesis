import logging
import random
import os
from pathlib import Path
import argparse
import pickle

import numpy as np

from sklearn.metrics import ConfusionMatrixDisplay 

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

import transformers
from transformers import AutoTokenizer, AutoModel
from transformers import DataCollatorWithPadding
from transformers import get_scheduler

import datasets
from datasets import load_from_disk, load_metric, DatasetDict

from tqdm.auto import tqdm

from accelerate import Accelerator


from metrics import *
from visualizations import *





# Argparser -----------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune NoExp or ExpBERT"+\
    " model on natural disaster tweet classification task")

    # Flags --------------------------------------------------------------
    parser.add_argument(
        '--exp-flag', 
        action='store_true', 
        help="Run ExpBERT"
    )

    parser.add_argument(
        "--full-logger",
        action="store_true",
        help="Set all logger settings on (including library loggers).",
    )
    
    parser.add_argument(
        "--tiny-dataset",
        action="store_true",
        help="Use smaller dataset for training and evaluation.",
    )

    parser.add_argument(
        "--use-saved-model",
        action="store_true",
        help="Use model that you have saved from previous run.",
    )

    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save the model that you will train.",
    )

    parser.add_argument(
        '--eval', 
        action='store_true'
    )

    # Directories -------------------------------------------------------------
    parser.add_argument(
        "--output-logs", 
        type=str, 
        default="logs", 
        help="Where to store the logs of the model."
    )
    
    parser.add_argument(
        "--output-metrics", 
        type=str, 
        default="metrics", 
        help="Where to store the metrics of the model during training and testing."
    )

    parser.add_argument(
        "--saved-model-filepath", 
        type=str, 
        default=None, 
        help="Location of the model that has been saved that will be used for training."
    )

    parser.add_argument(
        "--output-model", 
        type=str, 
        default="saved_model", 
        help="Location of where to save the model that will be trained."
    )

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

    # Others ------------------------------------------------------------------
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default="bert-base-cased", 
        help="Specify the checkpoint of your model e.g. bert-base-cased."
    )

    parser.add_argument(
        "--num-epochs", 
        type=int, 
        default=3, 
        help="Total number of epochs to perform during training."
    )

    parser.add_argument(
        "--percent-dataset", 
        type=float, 
        default=1.0, 
        help="Percentage of the training data to use."
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None, 
        help="A seed for dataset shuffling."
    )

    
    # Sanity checks ------------------------------------------------------

    args = parser.parse_args()
 
    if args.use_saved_model and (args.saved_model_filepath is None or
        os.path.isfile(args.saved_model_filepath)):
        raise ValueError("Need to provide the correct filepath of the" +\
        " saved model you want to load.")

    if args.save_model and args.output_model is None:
        raise ValueError("Must provide directory to save model that will" +\
        " be trained")

    if len(args.checkpoint) > 0 and args.use_saved_model:
        raise ValueError("Can't provide checkpoint and saved model directory")

    if args.eval and args.use_saved_model == False:
        raise ValueError("Can only evaluate a model that has been saved before.")

   

    return args


# Helper functions ------------------------------------------------------------
def get_explanation_type(exp_dataset_filepath):
    if exp_dataset_filepath == "./dataset/crisis_dataset/noexp/" or ("size" in
        exp_dataset_filepath):
        explanation_type = "normal"
    else:
        #e.g. ./dataset/crisis_dataset_few/exp/
        filename = exp_dataset_filepath.split("/")
        idx_explanation = [idx for idx, s in enumerate(filename) if 'crisis_dataset' in s][0]
        explanation_type = filename[idx_explanation].split("_")[-1]

    return explanation_type

# Summary writer helper functions --------------------------------------------
def get_filepath_numbered(log_dir, exp_flag, checkpoint, num_epochs,
    percent_dataset, explanation_type):
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.
    """ 
    checkpoint = checkpoint.replace("-","_") 
    if exp_flag:
        tb_log_dir_prefix = (
            f"Exp_{checkpoint}_" 
            f"pd={percent_dataset}_" 
            f"epochs={num_epochs}_" 
            f"explanations={explanation_type}_"
            f"run_"
            )
    else:
        tb_log_dir_prefix = (
            f"NoExp_{checkpoint}_"
            f"pd={percent_dataset}_" 
            f"epochs={num_epochs}_" 
            f"run_"
            )

    i = 0
    while i < 1000:
        #Creates the PosixPath with run iteration appended
        tb_log_dir = log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)



# Tokenizer functions -------------------------------------------------

def decode_text(tokenizer, text, exp_flag, *args):
    if exp_flag:
        encoded_input = tokenizer(text, args[0]) #args[0]=explanations
    else:
        encoded_input = tokenizer(text)

    decoded_text = tokenizer.decode(encoded_input["input_ids"])
    return decoded_text


# Smaller datasets to speed up training ----------------------------

def create_tiny_dataset(tokenized_datasets, seed):
    small_train_dataset = \
        tokenized_datasets["train"].shuffle(seed).select(range(80))
    small_test_dataset = \
        tokenized_datasets["test"].shuffle(seed).select(range(20))

    small_datasets = DatasetDict({"train": small_train_dataset, "test": small_test_dataset})
    return small_datasets


# Main -------------------------------------------------------------------


def main():
    args = parse_args()

    # Logger and summary writer --------------------------------------------
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
   

    #Find explanation type (normal, bad, few, many)
    explanation_type = get_explanation_type(args.exp_dataset_filepath)
    
    if args.checkpoint == "cardiffnlp/twitter-roberta-base":
        checkpoint = args.checkpoint.split("/")[-1]
        logs_filepath = get_filepath_numbered(Path(args.output_logs), args.exp_flag,
            checkpoint, args.num_epochs, args.percent_dataset,
            explanation_type)
    else:
        logs_filepath = get_filepath_numbered(Path(args.output_logs), args.exp_flag,
            args.checkpoint, args.num_epochs, args.percent_dataset,
            explanation_type)

    if not os.path.exists('metrics'):
        os.makedirs('metrics')

    
    #current run is the name used for all visualizations for a specific run
    current_run = logs_filepath.split("/")[-1]
    current_run_number = int(current_run.split("_")[-1])

    metrics_filepath = "./metrics/" + current_run + "/"
    plots_filepath = "./plots/" + current_run + "/"

    if not os.path.exists(metrics_filepath):
        os.makedirs(metrics_filepath)

    if not os.path.exists(plots_filepath):
        os.makedirs(plots_filepath)

    summary_writer = SummaryWriter(str(logs_filepath), flush_secs=5)

    accelerator = Accelerator()

    logger.info(accelerator.state)

    args.full_logger = False

    if args.full_logger:
        # Setup logging, we only want one process per machine to log things on the
        # screen.
        # accelerator.is_local_main_process is only True for one process per machine.

        logger.setLevel(logging.INFO if accelerator.is_local_main_process 
            else logging.ERROR)

        if accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_info() #outputs model config
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()



    # Important variables and flags --------------------------------------------
    # TODO: turn variables into arguments passed from calling program

    batch_size=8
    
    #Use latest model that was saved
    #Need to shift run number by -1 as latest model that has been trained is
    #current run - 1
    output_directory_save_model = get_filepath_numbered(Path(args.output_model),
        args.exp_flag, args.checkpoint, args.num_epochs, args.percent_dataset, 
        explanation_type)

    #Model is set to evaluation mode by default using model.eval()
    #Using checkpoint is much quicker as model and tokenizer are cached by Huggingface
    if args.use_saved_model:
        model = \
        AutoModel.from_pretrained(args.saved_model_filepath,
            num_labels=9)
        tokenizer = AutoTokenizer.from_pretrained(model_saved_filepath)
    else:
        model = AutoModel.from_pretrained(args.checkpoint,
            num_labels=9)
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

    # Loading dataset ---------------------------------------------
    if args.exp_flag:
        raw_datasets = load_from_disk(args.exp_dataset_filepath)
    else:
        raw_datasets = load_from_disk(args.noexp_dataset_filepath)
        #raw_datasets = raw_datasets["train"].train_test_split(train_size=0.8,
        #    shuffle=True)
    
    print(raw_datasets)
    
    # Get embeddings ------------------------------------------------------

    #TODO: Make this section compatible with dataset percent
    if args.tiny_dataset:
        tokenized_train = tokenizer(raw_datasets['train']['text'][:15], truncation=True, padding=True, return_tensors='pt')
    else:
        tokenized_train = tokenizer(raw_datasets['train']['text'], truncation=True, padding=True, return_tensors='pt')


    train_ids = tokenized_train['input_ids']
    model_outputs = model(train_ids)
    
    #Embeddings is of dimensions number of tokens x 768 (output layer of BERT)
    output = model_outputs['last_hidden_state'] #0 is the CLS token
    
    #Obtain embeddings for all datapoints (num datapoints X 768)
    #768 is the number of output neurons in final layer of BERT
    embeddings = output[:,0,:]
    print(embeddings.shape)

    if args.exp_flag:
       #Want to convert (670760x768) into (18660x1x768)
       #split, then restack
       pass


    #TODO: concatenate the embeddings for classifier
    #shape: (num_explanations + num textual_descriptions) x 768
    
    #Save embedding as pickle file 
    #torch.save(embeddings, 'embeddings.pt')
    
    #To load the embeddings
    #torch.load('tensors.pt')

   


if __name__ == "__main__":
    main()
