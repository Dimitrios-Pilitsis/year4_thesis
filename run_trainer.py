from transformers import AutoTokenizer, AutoModelForSequenceClassification

from datasets import load_dataset

from transformers import DataCollatorWithPadding
from datasets import load_metric
from transformers import TrainingArguments
from transformers import Trainer
# Basic variables --------------------------------------------


output_directory = "./saved_model/"



# Loading dataset ---------------------------------------------

raw_datasets = load_dataset("glue", "mrpc")

checkpoint = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint,
num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Helper functions --------------------------------------------
def save_model_tokenizer(directory):
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)




# Tokenizers ----------------------------------------------------
#Is dataset specific
def tokenize_function(examples):
    #return tokenizer(examples["text"], truncation=True)
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)



#Collator function for padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


print(raw_datasets)

# Metrics -----------------------------------------------------


metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


#Fine tuning-----------------------------------------------------

training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch",
report_to = "none")

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

# Saving model --------------------------------------------------

def save_model(output_directory):
    model.save_pretrained(output_directory)
    tokenizer.save_pretrained(output_directory)

#save_model(output_directory)
