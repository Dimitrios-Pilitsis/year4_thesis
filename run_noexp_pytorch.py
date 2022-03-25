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
def get_summary_writer_log_dir(log_dir) -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.
    """ 
    tb_log_dir_prefix = (
          f"NoExp_"
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



# Tokeinzer functions -------------------------------------------------

def decode_text(tokenizer, text):
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

    model.eval()

    
    for batch in dataloader:
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        
        metric1.add_batch(predictions=accelerator.gather(predictions),
            references=accelerator.gather(batch['labels']))
        metric2.add_batch(predictions=accelerator.gather(predictions),
            references=accelerator.gather(batch['labels']))
        metric3.add_batch(predictions=accelerator.gather(predictions),
            references=accelerator.gather(batch['labels']))
        metric4.add_batch(predictions=accelerator.gather(predictions),
            references=accelerator.gather(batch['labels']))

        metric2_weighted.add_batch(predictions=accelerator.gather(predictions),
            references=accelerator.gather(batch['labels']))
        metric3_weighted.add_batch(predictions=accelerator.gather(predictions),
            references=accelerator.gather(batch['labels']))
        metric4_weighted.add_batch(predictions=accelerator.gather(predictions),
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
     

    summary_writer.add_scalars("Train f1 per class",
                              {"f1_class_0/train": train_metrics["f1"][0],
                              "f1_class_1/train" : train_metrics["f1"][1],
                              "f1_class_2/train" : train_metrics["f1"][2],
                              "f1_class_3/train" : train_metrics["f1"][3],
                              "f1_class_4/train" : train_metrics["f1"][4],
                              "f1_class_5/train" : train_metrics["f1"][5],
                              "f1_class_6/train" : train_metrics["f1"][6],
                              "f1_class_7/train" : train_metrics["f1"][7],
                              "f1_class_8/train" : train_metrics["f1"][8]},
                              epoch+1)


    summary_writer.add_scalars("Train precision per class",
                       {"precision_class_0/train" : train_metrics["precision"][0],
                        "precision_class_1/train" : train_metrics["precision"][1],
                        "precision_class_2/train" : train_metrics["precision"][2],
                        "precision_class_3/train" : train_metrics["precision"][3],
                        "precision_class_4/train" : train_metrics["precision"][4],
                        "precision_class_5/train" : train_metrics["precision"][5],
                        "precision_class_6/train" : train_metrics["precision"][6],
                        "precision_class_7/train" : train_metrics["precision"][7],
                        "precision_class_8/train" : train_metrics["precision"][8]},
                        epoch+1)
    

    summary_writer.add_scalars("Train recall per class",
                                {"recall_class_0/train" : train_metrics["recall"][0],
                                 "recall_class_1/train" : train_metrics["recall"][1],
                                 "recall_class_2/train" : train_metrics["recall"][2],
                                 "recall_class_3/train" : train_metrics["recall"][3],
                                 "recall_class_4/train" : train_metrics["recall"][4],
                                 "recall_class_5/train" : train_metrics["recall"][5],
                                 "recall_class_6/train" : train_metrics["recall"][6],
                                 "recall_class_7/train" : train_metrics["recall"][7],
                                 "recall_class_8/train" : train_metrics["recall"][8]},
                                 epoch+1)



def summary_writer_test(summary_writer, test_metrics, epoch):
    summary_writer.add_scalars("Test averaged statistics",
                                 {"accuracy/test": test_metrics["accuracy"],
                                  "f1_weighted/test": test_metrics["f1_weighted"],
                                  "precision_weighted/test": test_metrics["precision_weighted"],
                                  "recall_weighted/test": test_metrics["recall_weighted"]},
                                  epoch+1)


    summary_writer.add_scalars("Test f1 per class",
                              {"f1_class_0/test": test_metrics["f1"][0],
                              "f1_class_1/test" : test_metrics["f1"][1],
                              "f1_class_2/test" : test_metrics["f1"][2],
                              "f1_class_3/test" : test_metrics["f1"][3],
                              "f1_class_4/test" : test_metrics["f1"][4],
                              "f1_class_5/test" : test_metrics["f1"][5],
                              "f1_class_6/test" : test_metrics["f1"][6],
                              "f1_class_7/test" : test_metrics["f1"][7],
                              "f1_class_8/test" : test_metrics["f1"][8]},
                              epoch+1)


    summary_writer.add_scalars("Test precision per class",
                       {"precision_class_0/test" : test_metrics["precision"][0],
                        "precision_class_1/test" : test_metrics["precision"][1],
                        "precision_class_2/test" : test_metrics["precision"][2],
                        "precision_class_3/test" : test_metrics["precision"][3],
                        "precision_class_4/test" : test_metrics["precision"][4],
                        "precision_class_5/test" : test_metrics["precision"][5],
                        "precision_class_6/test" : test_metrics["precision"][6],
                        "precision_class_7/test" : test_metrics["precision"][7],
                        "precision_class_8/test" : test_metrics["precision"][8]},
                        epoch+1)
   

    summary_writer.add_scalars("Test recall per class",
                            {"recall_class_0/test" : test_metrics["recall"][0],
                             "recall_class_1/test" : test_metrics["recall"][1],
                             "recall_class_2/test" : test_metrics["recall"][2],
                             "recall_class_3/test" : test_metrics["recall"][3],
                             "recall_class_4/test" : test_metrics["recall"][4],
                             "recall_class_5/test" : test_metrics["recall"][5],
                             "recall_class_6/test" : test_metrics["recall"][6],
                             "recall_class_7/test" : test_metrics["recall"][7],
                             "recall_class_8/test" : test_metrics["recall"][8]},
                             epoch+1)


# Visualizations ---------------------------------------------

def create_confusion_matrix(labels, predictions):
    ConfusionMatrixDisplay.from_predictions(predictions, labels) 

    plt.savefig('./plots/confusion_matrix.png', bbox_inches='tight')
    plt.show()


def visualizations(accelerator, model, dataloader):
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
       
    
    labels = np.array(labels).flatten()
    predictions = np.array(predictions).flatten()

    print(labels)
    print(predictions)
    
    # Individual visualizations
    create_confusion_matrix(labels, predictions)




# Saving model --------------------------------------------------

def save_model(output_directory_save_model):
    model.save_pretrained(output_directory_save_model)
    tokenizer.save_pretrained(output_directory_save_model)



# Main -------------------------------------------------------------------


def main():
    # Logger and summary writer --------------------------------------------
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logs_directory = get_summary_writer_log_dir(Path('logs'))

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

    num_epochs = 3
    batch_size=8

    output_directory_save_model = "./saved_model/"

    checkpoint = "bert-base-cased"

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint,
        num_labels=9)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)


    # Loading dataset ---------------------------------------------

    raw_datasets = load_from_disk("./dataset/crisis_dataset/noexp/")

    # Log a few random samples from the training set:
    for index in random.sample(range(len(raw_datasets['train'])), 6):
        logger.info(f"Data point {index} of the training set sample: {raw_datasets['train'][index]['text']}.")
        logger.info(f"Data point {index} of the training set sample: {raw_datasets['train'][index]['labels']}.")


    # Tokenizers ----------------------------------------------------
    #Is dataset specific
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True)


    def tokenize_function_with_explanation(examples):
        return tokenizer(examples['text'], examples['explanation'],
            truncation=True)


    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)


    #Remove columns that aren't strings here
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])


    #Collator function for padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Log a few random samples from the tokenized dataset:
    for index in random.sample(range(len(tokenized_datasets['train'])), 6):
        logger.info(f"Data point {index} of the tokenized training set sample: {decode_text(tokenizer, raw_datasets['train'][index]['text'])}.")

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
        visualizations(accelerator, model, train_dataloader)
        exit(0)
        # Testing ----------------------------------------------------------------------
        logger.info(f"***** Running test set Epoch {epoch + 1}*****")
        test_metrics = compute_metrics(accelerator, model, test_dataloader)
        logger.info(f"Epoch {epoch + 1} Test results: {test_metrics}")
        
        summary_writer_test(summary_writer, test_metrics, epoch)
    

    # Plots for final model parameters ------------------------------------------------
    #visualizations(accelerator, model, test_dataloader)


    summary_writer.close()

    #save_model(output_directory_save_model)


if __name__ == "__main__":
    main()
