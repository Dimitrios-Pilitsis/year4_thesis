from datasets import load_from_disk, load_metric

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments
from transformers import Trainer


import numpy as np



# Basic variables --------------------------------------------


output_directory_save_model = "./saved_model/"

checkpoint = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint,
num_labels=2)

#model = AutoModelForSequenceClassification.from_pretrained(checkpoint,
#num_labels=9

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Loading dataset ---------------------------------------------

raw_datasets = load_from_disk("./dataset/crisis_dataset")

# NEED TO SHUFFLE DATASET



def check_dataset(dataset):
    print(dataset)
    print(dataset['train'].shuffle(42).select(range(3))['text'])

#check_dataset(raw_datasets)


#raw_datasets = load_dataset("glue", "mrpc")




# Tokenizers ----------------------------------------------------
#Is dataset specific
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)
    #return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)


#Remove columns that aren't strings here
#tokenized_datasets = tokenized_datasets.remove_columns(["idx", "sentence1",
#"sentence2"])

tokenized_datasets = tokenized_datasets.remove_columns(["text"])


#Collator function for padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)



def check_tokenized_dataset(tokenizer, tokenized_datasets):
    print(tokenized_datasets)
    decoded = tokenizer.decode(tokenized_datasets['train']["input_ids"][0])
    print(decoded)


check_tokenized_dataset(tokenizer, tokenized_datasets)


# Metrics -----------------------------------------------------


#metric = load_metric("accuracy")
metric = load_metric("accuracy", "f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


#Fine tuning-----------------------------------------------------

training_args = TrainingArguments("./test-trainer/", evaluation_strategy="epoch",
report_to = "none")


#training_args = TrainingArguments("./test-trainer/", evaluation_strategy="steps",
#eval_steps=2, report_to="none")



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

