import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import get_scheduler

from datasets import load_from_disk, load_metric, DatasetDict

from tqdm.auto import tqdm



# Important variables and flags --------------------------------------------
# TODO: turn variables into arguments passed from calling program
flag_smaller_datasets = True
flag_check_dataset = False
flag_check_tokenized_dataset = False

num_epochs = 3

output_directory_save_model = "./saved_model/"

checkpoint = "bert-base-cased"

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




# Tokenizer helper functions ---------------------------------------
def check_tokenized_dataset(tokenizer, tokenized_datasets):
    print(tokenized_datasets)
    decoded = tokenizer.decode(tokenized_datasets['train']["input_ids"][0])
    print(decoded)

if flag_check_tokenized_dataset:
    check_tokenized_dataset(tokenizer, tokenized_datasets)




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
    print(decoded_text)



decode_text(tokenizer, "BREAKING NEWS 400 #Earthquakes in #SanFrancisco.")
decode_text(tokenizer, "RT @heyyouapp: California USA Rancho Cucamonga http://t.co/TPSViMWZMs BREAKING NEWS 400 Earthquake Powerful Earthquake Slams San Francisco...")
decode_text(tokenizer, "(♡Alixandro Wilson♡) Renewed Calls for Early-Warning System After Quake: California q... http://t.co/Nx46sLwrDQ (♡Alixandro Wilson♡)")
decode_text(tokenizer, "RT @janinebucks: ♦ http://t.co/sxEjExYuEM 927 ♦ earthquake ♦ Damage From Northern California Earthquake Could Reach $1 Billio nhttp://t.co/… ")
decode_text(tokenizer, "RT @realsyedasarwat: May ALLAH give them strength who's family/relatives dead in #earthquake my prayers r with them! 😔 ")



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
    batch_size=8, collate_fn=data_collator)
eval_dataloader = DataLoader(tokenized_datasets['validation'], batch_size=8, 
    collate_fn=data_collator)
test_dataloader = DataLoader(tokenized_datasets['test'], batch_size=8, 
    collate_fn=data_collator)



# Optimizer and learning rate scheduler -----------------------------


optimizer = AdamW(model.parameters(), lr=5e-5)

num_training_steps = num_epochs * len(train_dataloader)

lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0,
    num_training_steps=num_training_steps
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model.to(device)



# Metrics -----------------------------------------------------

def compute_metrics(model, dataloader):
    metric1 = load_metric("accuracy")
    metric2 = load_metric("f1")

    model.eval()

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        
        accuracy = metric1.compute(predictions=predictions,
            references=batch["labels"])["accuracy"]
        f1_weighted = metric2.compute(predictions=predictions,
            references=batch["labels"], average="weighted")["f1"]
        f1_macro = metric2.compute(predictions=predictions,
            references=batch["labels"], average="macro")["f1"]
        return {"accuracy": accuracy, "f1_weighted": f1_weighted, "f1_macro":
        f1_macro}



#Training loop -------------------------------------------------------------


progress_bar = tqdm(range(num_training_steps))

model.train()

for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    train_metrics = compute_metrics(model, train_dataloader)
    print(f'Epoch {epoch + 1} results')
    print(train_metrics)




# Evaluation and testing -------------------------

print("Evaluation results")
eval_metrics = compute_metrics(model, eval_dataloader)
print(eval_metrics)


print("Test results")
test_metrics = compute_metrics(model, test_dataloader)
print(test_metrics)




# Saving model --------------------------------------------------

def save_model(output_directory_save_model):
    model.save_pretrained(output_directory_save_model)
    tokenizer.save_pretrained(output_directory_save_model)

#save_model(output_directory_save_model)

