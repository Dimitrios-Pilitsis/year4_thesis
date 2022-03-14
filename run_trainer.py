from datasets import load_from_disk, load_metric, DatasetDict

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments
from transformers import Trainer


import numpy as np



# Important variables and flags --------------------------------------------
# TODO: turn variables into arguments passed from calling program
flag_smaller_datasets = True
flag_check_dataset = False
flag_check_tokenized_dataset = False
flag_evaluation_strategy = "epoch" #alternative is steps

output_directory_save_model = "./saved_model/"

checkpoint = "bert-base-uncased"

model = AutoModelForSequenceClassification.from_pretrained(checkpoint,
num_labels=8)

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Loading dataset ---------------------------------------------

raw_datasets = load_from_disk("./dataset/crisis_dataset")


def check_dataset(dataset):
    print(dataset)
    print(dataset['train'].shuffle(42).select(range(3))['text'])

if flag_check_dataset:
    check_dataset(raw_datasets)






# Tokenizers ----------------------------------------------------
#Is dataset specific
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)


#Remove columns that aren't strings here
tokenized_datasets = tokenized_datasets.remove_columns(["text"])


#Collator function for padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# Tokenizer helper functions ---------------------------------------
def check_tokenized_dataset(tokenizer, tokenized_datasets):
    print(tokenized_datasets)
    decoded = tokenizer.decode(tokenized_datasets['train']["input_ids"][0])
    print(decoded)

if flag_check_tokenized_dataset:
    check_tokenized_dataset(tokenizer, tokenized_datasets)


# Smaller datasets to speed up training ----------------------------
def create_smaller_dataset(tokenized_datasets):
    small_train_dataset = \
        tokenized_datasets["train"].shuffle(seed=42).select(range(800))
    small_validation_dataset = \
        tokenized_datasets["validation"].shuffle(seed=42).select(range(100))
    small_test_dataset = \
        tokenized_datasets["test"].shuffle(seed=42).select(range(100))

    small_datasets = DatasetDict({"train": small_train_dataset, 
        "validation": small_validation_dataset, "test":small_test_dataset})
    return small_datasets

if flag_smaller_datasets:
    tokenized_datasets = create_smaller_dataset(tokenized_datasets)






# Metrics -----------------------------------------------------


#metric = load_metric("accuracy")
metric = load_metric("accuracy", "f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


#Fine tuning-----------------------------------------------------

if flag_evaluation_strategy == "epoch":
    training_args = TrainingArguments("./test-trainer/", 
        evaluation_strategy="epoch", report_to = "none")
else:
    training_args = TrainingArguments("./test-trainer/", 
        evaluation_strategy="steps", eval_steps=50, report_to="none")



trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

# Prediction ---------------------------------------------------


predictions = trainer.predict(tokenized_datasets["test"])
print(predictions.predictions.shape, predictions.label_ids.shape)
metric.compute(predictions)



# Saving model --------------------------------------------------

def save_model(output_directory_save_model):
    model.save_pretrained(output_directory_save_model)
    tokenizer.save_pretrained(output_directory_save_model)

#save_model(output_directory_save_model)

