import logging
import random
import os
from pathlib import Path


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

# Logger ---------------------------------------------------------------

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
    num_labels=8)

tokenizer = AutoTokenizer.from_pretrained(checkpoint)


# Loading dataset ---------------------------------------------

raw_datasets = load_from_disk("./dataset/crisis_dataset")


# Log a few random samples from the training set:
for index in random.sample(range(len(raw_datasets)), 3):
    logger.info(f"Data point {index} of the training set sample: {raw_datasets['train'][index]['text']}.")


# Tokenizers ----------------------------------------------------
#Is dataset specific
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)


#Remove columns that aren't strings here
tokenized_datasets = tokenized_datasets.remove_columns(["text"])


#Collator function for padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


def decode_text(tokenizer, text):
    encoded_input = tokenizer(text)
    decoded_text = tokenizer.decode(encoded_input["input_ids"])
    return decoded_text


# Log a few random samples from the tokenized dataset:
for index in random.sample(range(len(tokenized_datasets)), 3):
    logger.info(f"Data point {index} of the tokenized training set sample: {decode_text(tokenizer, raw_datasets['train'][index]['text'])}.")



# Smaller datasets to speed up training ----------------------------
def create_smaller_dataset(tokenized_datasets):
    small_train_dataset = \
        tokenized_datasets["train"].shuffle(seed=42).select(range(80))
    small_validation_dataset = \
        tokenized_datasets["validation"].shuffle(seed=42).select(range(10))
    small_test_dataset = \
        tokenized_datasets["test"].shuffle(seed=42).select(range(10))

    small_datasets = DatasetDict({"train": small_train_dataset, 
        "validation": small_validation_dataset, "test": small_test_dataset})
    return small_datasets

if flag_smaller_datasets:
    tokenized_datasets = create_smaller_dataset(tokenized_datasets)


tokenized_datasets.set_format("torch")



# Dataloader --------------------------------------------------

train_dataloader = DataLoader(tokenized_datasets['train'], shuffle=True, 
    batch_size=batch_size, collate_fn=data_collator)
eval_dataloader = DataLoader(tokenized_datasets['validation'],
    batch_size=batch_size, collate_fn=data_collator)
test_dataloader = DataLoader(tokenized_datasets['test'], batch_size=batch_size, 
    collate_fn=data_collator)


# Optimizer and learning rate scheduler -----------------------------

optimizer = AdamW(model.parameters(), lr=5e-5)

train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer
)

num_training_steps = num_epochs * len(train_dataloader)

lr_scheduler = get_scheduler(
    name="linear", 
    optimizer=optimizer, 
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)



# Metrics -----------------------------------------------------

def compute_metrics(model, dataloader):
    metric1 = load_metric("accuracy")
    metric2 = load_metric("f1")
    metric3 = load_metric("f1")

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


    accuracy = metric1.compute()["accuracy"]
    f1_weighted = metric2.compute(average="weighted")["f1"]
    f1_macro = metric3.compute(average="macro")["f1"]

    results = {"accuracy": accuracy, "f1_weighted": f1_weighted,
        "f1_macro": f1_macro}
    return results

        



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


    train_metrics = compute_metrics(model, train_dataloader)
    logger.info(f"Epoch {epoch + 1} results: {train_metrics}")
    summary_writer.add_scalar('accuracy/train', train_metrics['accuracy'], epoch)
    summary_writer.add_scalar('f1_weighted/train', train_metrics['f1_weighted'], epoch)
    summary_writer.add_scalar('f1_macro/train', train_metrics['f1_macro'], epoch)




# Evaluation and testing -------------------------
logger.info("***** Running evaluation set *****")
eval_metrics = compute_metrics(model, eval_dataloader)
logger.info(f"Evalation results: {eval_metrics}")


logger.info("***** Running test set *****")
test_metrics = compute_metrics(model, test_dataloader)
logger.info(f"Test results: {test_metrics}")
summary_writer.add_scalar('accuracy/eval', test_metrics['accuracy'], epoch)
summary_writer.add_scalar('f1_weighted/eval', test_metrics['f1_weighted'], epoch)
summary_writer.add_scalar('f1_macro/eval', test_metrics['f1_macro'], epoch)


summary_writer.close()

# Saving model --------------------------------------------------

def save_model(output_directory_save_model):
    model.save_pretrained(output_directory_save_model)
    tokenizer.save_pretrained(output_directory_save_model)

#save_model(output_directory_save_model)


