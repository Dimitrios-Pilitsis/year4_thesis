import logging
import random
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

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



# Summary writer helper functions --------------------------------------------
def get_dir_numbered(log_dir, exp_flag):
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.
    """ 
    
    if exp_flag:
        tb_log_dir_prefix = (f"Exp_run_")
    else:
        tb_log_dir_prefix = (f"NoExp_run_")

    i = 0
    while i < 1000:
        #Creates the PosixPath with run iteration appended
        tb_log_dir = log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)

def get_saved_model_dir(saved_model_dir, exp_flag):
    if exp_flag:
        saved_model_prefix = (f"Exp_run_")
    else:
        saved_model_prefix = (f"NoExp_run_")

    i=0
    while i < 1000:
        #Creates the PosixPath with run iteration appended
        tb_log_dir = log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)

    model_saved_directory = "./saved_model/" + current_run.replace(current_run[-1], str(current_run_number-1))

# Tokenizer functions -------------------------------------------------

def decode_text(tokenizer, text, exp_flag, *args):
    if exp_flag:
        encoded_input = tokenizer(text, args[0]) #args[0]=explanations
    else:
        encoded_input = tokenizer(text)

    decoded_text = tokenizer.decode(encoded_input["input_ids"])
    return decoded_text


# Smaller datasets to speed up training ----------------------------
def create_smaller_dataset(tokenized_datasets):
    small_train_dataset = \
        tokenized_datasets["train"].shuffle(seed=42).select(range(80))
    small_test_dataset = \
        tokenized_datasets["test"].shuffle(seed=42).select(range(20))

    small_datasets = DatasetDict({"train": small_train_dataset, "test": small_test_dataset})
    return small_datasets




# Metrics -----------------------------------------------------

def compute_metrics(accelerator, model, dataloader):

    metric1 = load_metric("accuracy")
    metric2 = load_metric("f1")
    metric3 = load_metric("precision")
    metric4 = load_metric("recall")
    metric2_weighted = load_metric("f1")
    metric3_weighted = load_metric("precision")
    metric4_weighted  = load_metric("recall")

    metrics = [metric1, metric2, metric3, metric4, metric2_weighted,
        metric3_weighted, metric4_weighted]

    model.eval()

    for batch in dataloader:
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        for metric in metrics:
            metric.add_batch(predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch['labels']))

    #Per class metrics
    labels = list(range(0,9))
    f1 = metric2.compute(average=None, labels=labels)["f1"]
    precision = metric3.compute(average=None, labels=labels)["precision"]
    recall = metric4.compute(average=None, labels=labels)["recall"]


    #Weighted metrics
    accuracy = metric1.compute()["accuracy"]
    f1_weighted = metric2_weighted.compute(average="weighted", labels=labels)["f1"]
    precision_weighted = metric3_weighted.compute(average="weighted", labels=labels)["precision"]
    recall_weighted = metric4_weighted.compute(average="weighted", labels=labels)["recall"]

    results = {"accuracy": accuracy, 
                "f1_weighted": f1_weighted,
                "precision_weighted": precision_weighted,
                "recall_weighted": recall_weighted,
                "f1": f1,
                "precision": precision,
                "recall": recall
            }

    return results




# Summary writers

def summary_writer_train(summary_writer, train_metrics, epoch):
    summary_writer.add_scalars("Train averaged statistics",
                                    {"accuracy/train": train_metrics["accuracy"],
                                     "f1_weighted/train": train_metrics["f1_weighted"],
                                     "precision_weighted/train": train_metrics["precision_weighted"],
                                     "recall_weighted/train":
                                     train_metrics["recall_weighted"]},
                                     epoch+1)
     

    f1_train = {}
    precision_train = {}
    recall_train = {}

    for idx, val in enumerate(train_metrics["f1"]):
        f1_train[f'f1_class_{idx}/train'] = val
    
    for idx, val in enumerate(train_metrics["precision"]):
        precision_train[f'precision_class_{idx}/train'] = val
    
    for idx, val in enumerate(train_metrics["recall"]):
        recall_train[f'recall_class_{idx}/train'] = val

    summary_writer.add_scalars("Train f1 per class", f1_train, epoch+1)

    summary_writer.add_scalars("Train precision per class", precision_train,
        epoch+1)

    summary_writer.add_scalars("Train recall per class", recall_train, epoch+1)
                               



def summary_writer_test(summary_writer, test_metrics, epoch):

    summary_writer.add_scalars("Test averaged statistics",
                                 {"accuracy/test": test_metrics["accuracy"],
                                  "f1_weighted/test": test_metrics["f1_weighted"],
                                  "precision_weighted/test": test_metrics["precision_weighted"],
                                  "recall_weighted/test": test_metrics["recall_weighted"]},
                                  epoch+1)

    f1_test = {}
    precision_test = {}
    recall_test = {}

    for idx, val in enumerate(test_metrics["f1"]):
        f1_test[f'f1_class_{idx}/test'] = val
    
    for idx, val in enumerate(test_metrics["precision"]):
        precision_test[f'precision_class_{idx}/test'] = val
    
    for idx, val in enumerate(test_metrics["recall"]):
        recall_test[f'recall_class_{idx}/test'] = val

    summary_writer.add_scalars("Test f1 per class", f1_test, epoch+1)

    summary_writer.add_scalars("Test precision per class", precision_test,
        epoch+1)

    summary_writer.add_scalars("Test recall per class", recall_test, epoch+1)




# Visualizations ---------------------------------------------

def create_confusion_matrix(labels, predictions, current_run):
    ConfusionMatrixDisplay.from_predictions(predictions, labels,
        labels=list(range(0,9)))
    
    filepath = "./plots/" + current_run + "_confusion_matrix.png"
    plt.savefig(filepath, bbox_inches='tight')
    plt.show()


def visualizations(accelerator, model, dataloader, current_run):
    labels = []
    predictions = []
    model.eval()

    for batch in dataloader:
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1)
        prediction = accelerator.gather(prediction).detach().cpu().numpy()
        label = accelerator.gather(batch['labels']).detach().cpu().numpy()
        labels.append(label)
        predictions.append(prediction)
       
    #Flattens array even when subarrays aren't of equal dimension (due to batch
    # size)
    labels = np.hstack(np.array(labels, dtype=object))
    predictions = np.hstack(np.array(predictions, dtype=object))

    print(labels)
    print(predictions)
    
    # Individual visualizations
    create_confusion_matrix(labels, predictions, current_run)



# Main -------------------------------------------------------------------


def main():

    exp_flag = True

    # Logger and summary writer --------------------------------------------
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logs_directory = get_dir_numbered(Path('logs'), exp_flag)
    current_run = logs_directory.split("/")[-1]
    current_run_number = int(current_run.split("_")[-1])

    summary_writer = SummaryWriter(str(logs_directory), flush_secs=5)

    accelerator = Accelerator()

    logger.info(accelerator.state)

    full_logger = False

    if full_logger:
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
    flag_smaller_datasets = True
    use_saved_model = False

    num_epochs = 3
    batch_size=8
    
    #Use latest model that was saved
    #Need to shift run number by -1 as latest model that has been trained is
    #current run - 1
    output_directory_save_model = get_dir_numbered(Path('saved_model'), exp_flag)

    model_saved_run_number = int(output_directory_save_model.split("_")[-1])-1
    model_saved_directory = \
        output_directory_save_model.replace(output_directory_save_model.split("_")[-1], \
        str(model_saved_run_number))

    #Do not allow model to be loaded if saved model is empty
    if not os.listdir(model_saved_directory.split("/")[0]):
        use_saved_model = False

    #Model is set to evaluation mode by default using model.eval()
    #Using checkpoint is much quicker as model and tokenizer are cached by Huggingface
    if use_saved_model:
        model = AutoModelForSequenceClassification.from_pretrained(model_saved_directory,
            num_labels=9)
        tokenizer = AutoTokenizer.from_pretrained(model_saved_directory)
    else:
        checkpoint = "bert-base-cased"
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint,
            num_labels=9)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)



    # Loading dataset ---------------------------------------------
    if exp_flag:
        raw_datasets = load_from_disk("./dataset/crisis_dataset/exp/")
    else:
        raw_datasets = load_from_disk("./dataset/crisis_dataset/noexp/")

    # Log a few random samples from the training set:
    for index in random.sample(range(len(raw_datasets['train'])), 4):
        logger.info(f"Text of data point {index} of the training set sample: {raw_datasets['train'][index]['text']}.")
        logger.info(f"Explanation of data point {index} of the training set "
                    f"example: {raw_datasets['train'][index]['exp_and_td']}.") if exp_flag else None
        logger.info(f"Label of data point {index} of the training set sample: {raw_datasets['train'][index]['labels']}.")


    # Tokenizers ----------------------------------------------------
    #Is dataset specific
    def tokenize_noexp_function(examples):
        return tokenizer(examples["text"], truncation=True)


    def tokenize_exp_function(examples):
        return tokenizer(examples['text'], examples['exp_and_td'],
            truncation=True)

    if exp_flag:
        tokenized_datasets = raw_datasets.map(tokenize_exp_function, batched=True)
        tokenized_datasets = tokenized_datasets.remove_columns(["exp_and_td"])
    else:
        tokenized_datasets = raw_datasets.map(tokenize_noexp_function, batched=True)


    #Remove columns that aren't strings here
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    print(tokenized_datasets)

    #Collator function for padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Log a few random samples from the tokenized dataset:
    for index in random.sample(range(len(tokenized_datasets['train'])), 4):

        if exp_flag:
            logger.info(f"Data point {index} of the tokenized training set sample: "
            f"{decode_text(tokenizer, raw_datasets['train'][index]['text'], {exp_flag}, raw_datasets['train'][index]['exp_and_td'])}.")
        else:
            logger.info(f"Data point {index} of the tokenized training set sample: "
            f"{decode_text(tokenizer, raw_datasets['train'][index]['text'],{exp_flag})}.")

        logger.info(f"Input IDs of data point {index} of the training set sample: {tokenized_datasets['train'][index]['input_ids']}.")
        logger.info(f"Token type IDs of data point {index} of the training set sample: {tokenized_datasets['train'][index]['token_type_ids']}.")

    
    if flag_smaller_datasets:
        tokenized_datasets = create_smaller_dataset(tokenized_datasets)

    tokenized_datasets.set_format("torch")



    # Dataloader --------------------------------------------------

    train_dataloader = DataLoader(tokenized_datasets['train'], shuffle=True, 
        batch_size=batch_size, collate_fn=data_collator)
    test_dataloader = DataLoader(tokenized_datasets['test'], batch_size=batch_size, 
        collate_fn=data_collator)


    # Optimizer and learning rate scheduler -----------------------------

    optimizer = AdamW(model.parameters(), lr=5e-5)

    train_dataloader, test_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader, test_dataloader, model, optimizer
    )

    num_training_steps = num_epochs * len(train_dataloader)

    lr_scheduler = get_scheduler(
        name="linear", 
        optimizer=optimizer, 
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )


    #Training loop -------------------------------------------------------------

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {tokenized_datasets['train'].num_rows}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(f"  Num training steps = {num_training_steps}")
    logger.info(f"  Instantaneous batch size per device = {batch_size}")
    progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_local_main_process)

    for epoch in range(num_epochs):
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
        logger.info(f"Epoch {epoch + 1} train results: {train_metrics}")
        summary_writer_train(summary_writer, train_metrics, epoch)

        #Visualizations so I don't wait for entire model to train
        #visualizations(accelerator, model, train_dataloader, current_run)
        #exit(0)
        # Testing ----------------------------------------------------------------------
        logger.info(f"***** Running test set Epoch {epoch + 1}*****")
        test_metrics = compute_metrics(accelerator, model, test_dataloader)
        logger.info(f"Epoch {epoch + 1} Test results: {test_metrics}")
        
        summary_writer_test(summary_writer, test_metrics, epoch)
    

    # Plots for final model parameters ------------------------------------------------
    visualizations(accelerator, model, test_dataloader, current_run)


    summary_writer.close()
    

    # Save model and tokenizer---------------------------------------------
    save_model_tokenizer = False
    if save_model_tokenizer:
        model.save_pretrained(output_directory_save_model)
        tokenizer.save_pretrained(output_directory_save_model)


if __name__ == "__main__":
    main()
