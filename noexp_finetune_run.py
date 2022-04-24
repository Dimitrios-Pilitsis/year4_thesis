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
from transformers import AutoTokenizer, AutoModelForSequenceClassification
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
        "--tiny-dataset",
        action="store_true",
        help="Use smaller dataset for training and evaluation.",
    )

    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save the model that you will train.",
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
        "--batch-size", 
        type=int, 
        default=8, 
        help="Batch size for model."
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
 
    if args.save_model and args.output_model is None:
        raise ValueError("Must provide directory to save model that will" +\
        " be trained")

    return args



# Summary writer helper functions --------------------------------------------
def get_filepath_numbered(log_dir, checkpoint, num_epochs, percent_dataset):
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.
    """ 
    checkpoint = checkpoint.replace("-","_") 
    tb_log_dir_prefix = (
        f"NoExpFinetune_{checkpoint}_"
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

def decode_text(tokenizer, text):
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

    if args.checkpoint == "cardiffnlp/twitter-roberta-base":
        checkpoint = args.checkpoint.split("/")[-1]
        logs_filepath = get_filepath_numbered(Path(args.output_logs), 
            checkpoint, args.num_epochs, args.percent_dataset)
    else:
        logs_filepath = get_filepath_numbered(Path(args.output_logs), 
            args.checkpoint, args.num_epochs, args.percent_dataset)

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


    # Important variables and flags --------------------------------------------
    #Need to shift run number by -1 as latest model that has been trained is
    #current run - 1
    output_directory_save_model = get_filepath_numbered(Path(args.output_model),
        args.checkpoint, args.num_epochs, args.percent_dataset)

    #Using checkpoint is much quicker as model and tokenizer are cached by Huggingface
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint,
        num_labels=9)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

    # Loading dataset ---------------------------------------------
    raw_datasets = load_from_disk(args.noexp_dataset_filepath)
    raw_datasets = raw_datasets["train"].train_test_split(train_size=0.8,
        shuffle=True)


    # Log a few random samples from the training set:
    for index in random.sample(range(len(raw_datasets['train'])), 4):
        logger.info(f"Text of data point {index} of the training set sample: {raw_datasets['train'][index]['text']}.")
        logger.info(f"Label of data point {index} of the training set sample: {raw_datasets['train'][index]['labels']}.")


    # Tokenizers ----------------------------------------------------
    #Is dataset specific
    def tokenize_noexp_function(examples):
        return tokenizer(examples["text"], truncation=True)

    tokenized_datasets = raw_datasets.map(tokenize_noexp_function, batched=True)

    #Remove columns that aren't strings here
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    print(tokenized_datasets)

    #Collator function for padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Log a few random samples from the tokenized dataset:
    for index in random.sample(range(len(tokenized_datasets['train'])), 4):

        logger.info(f"Data point {index} of the tokenized training set sample:"
            f"{decode_text(tokenizer, raw_datasets['train'][index]['text'])}.")

        logger.info(f"Input IDs of data point {index} of the training set sample: {tokenized_datasets['train'][index]['input_ids']}.")
   
    if args.tiny_dataset:
        tokenized_datasets = create_tiny_dataset(tokenized_datasets,
            args.seed)    

    tokenized_datasets.set_format("torch")


    # Dataloader --------------------------------------------------

    train_dataloader = DataLoader(tokenized_datasets['train'], shuffle=True, 
        batch_size=args.batch_size, collate_fn=data_collator)
    test_dataloader = DataLoader(tokenized_datasets['test'], shuffle=True,
        batch_size=args.batch_size, collate_fn=data_collator)


    # Optimizer and learning rate scheduler -----------------------------

    optimizer = AdamW(model.parameters(), lr=5e-5)

    train_dataloader, test_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader, test_dataloader, model, optimizer
    )

    num_training_steps = args.num_epochs * len(train_dataloader)

    lr_scheduler = get_scheduler(
        name="linear", 
        optimizer=optimizer, 
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )


    #Training loop -------------------------------------------------------------

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {tokenized_datasets['train'].num_rows}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Num training steps = {num_training_steps}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_local_main_process)

    train_metrics_list = []
    test_metrics_list = []


    for epoch in range(args.num_epochs):
        model.train()
        for batch in train_dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)


            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)


        train_metrics = compute_metrics(accelerator, model, train_dataloader)
        train_metrics_list.append(train_metrics)
        logger.info(f"Epoch {epoch + 1} train results: {train_metrics}")
        summary_writer_metrics(summary_writer, train_metrics, epoch,
            train_flag=True)
        
        # Testing ----------------------------------------------------------------------
        logger.info(f"***** Running test set Epoch {epoch + 1}*****")
        test_metrics = compute_metrics(accelerator, model, test_dataloader)
        test_metrics_list.append(test_metrics)
        logger.info(f"Epoch {epoch + 1} Test results: {test_metrics}")
        
        summary_writer_metrics(summary_writer, train_metrics, epoch,
            train_flag=False)

    # Plots for final model parameters ------------------------------------------------
    with open(metrics_filepath+'/train.p', 'wb') as fp:
        pickle.dump(train_metrics_list, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(metrics_filepath+'/test.p', 'wb') as fp:
        pickle.dump(test_metrics_list, fp, protocol=pickle.HIGHEST_PROTOCOL)

    visualizations_noexpfinetune(summary_writer, accelerator, model, test_dataloader, current_run, epoch)


    summary_writer.close()
    

    # Save model and tokenizer---------------------------------------------
    if args.save_model:
        model.save_pretrained(output_directory_save_model)
        tokenizer.save_pretrained(output_directory_save_model)


if __name__ == "__main__":
    main()
