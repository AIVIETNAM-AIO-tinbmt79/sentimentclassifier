import torch
from keras.src.metrics.accuracy_metrics import accuracy
from seqeval.metrics import accuracy_score

from dataset import SentimentDataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import re
import numpy as np


epochs = 3
max_length = 32
batch_size = 32
lr = 2e-5
model_name = "bert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_path = r"C:\NLP_roadmap\Transformer\training.1600000.processed.noemoticon.csv"

def load_data(file_path):
    data = pd.read_csv(file_path, encoding="latin-1", header = None)
    data.columns = ["target", "id", "date", "flag", "user", "text"]
    data = data.dropna()
    data = data.sample(300000, random_state=42)
    return data

data = load_data(file_path)


#preprocess
def preprocessing(text):
    text = re.sub("\s+", " ", text)
    text = text.lower()
    text = re.sub("@\w+", "", text)
    text = re.sub("http\S+", "", text)
    return text

### Calculate to choose a max_length ####
# text = data["text"].apply(preprocessing)
# length = [len(i.split()) for i in text]
# sum_length = sum(length)
# plt.figure(figsize = (10,10))
# plt.title(f"AVG LENGTH: {sum_length/ len(length):.2f}")
# plt.hist(length, bins = (0, 5, 10, 15, 20, 25, 30, 35, 40))
# plt.show()
tokenizer = AutoTokenizer.from_pretrained(model_name)
def prepare_data(df, tokenizer):
    text = df["text"].apply(preprocessing)
    target = df["target"].map({0:0, 4:1})
    train_texts, val_texts, train_labels, val_labels = train_test_split(text, target, test_size=0.1, random_state=42)

    train_encoding = tokenizer(
        train_texts.tolist(),
        padding="max_length",
        truncation=True,
        max_length=32,
    )

    val_encoding = tokenizer(
        val_texts.tolist(),
        padding="max_length",
        truncation=True,
        max_length=32,
    )

    train_dataset = SentimentDataset(train_encoding, train_labels.tolist())
    val_dataset = SentimentDataset(val_encoding, val_labels.tolist())
    return train_dataset, val_dataset

train_dataset, val_dataset = prepare_data(data, tokenizer)

##MODEL
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")

    return {
        "accuracy": acc,
        "f1": f1,
    }

training_args = TrainingArguments(
    output_dir=".\checkpoints",
    eval_strategy = "epoch",
    save_strategy = "epoch",

    learning_rate = lr,
    per_device_train_batch_size = batch_size,
    per_device_eval_batch_size = batch_size,

    num_train_epochs= epochs,
    weight_decay = 1e-2,

    logging_dir = ".\logs",
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1",

    report_to="none"
)

trainer = Trainer(
    model  = model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)


trainer.train()
metrics = trainer.evaluate()
print(metrics)
trainer.save_model("./sentiment_model")
tokenizer.save_pretrained("./sentiment_model")