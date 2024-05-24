from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer, 
)

from peft import peft_model, config
import evaluate
import torch
import numpy as np

model_checkpoint = 'distilbert-base-uncased'
id2label = {0: "Negative", 1: "Positive"}
label2id = {"Negative": 0, "Positive": 1}

model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=2, id2label=id2label, label2id=label2id)

dataset = load_dataset("shawhin/imdb-truncated")

tokenizer = AutoTokenizer.from_pretrained(
    model_checkpoint, add_prefix_space=True)


def tokenize_function(examples):
    text = examples['text']
    tokenizer.truncation_side = "left"
    tokenized_input = tokenizer(
        text=text, return_tensors="np", truncation=True, max_length=512)
    return tokenized_input


if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

tokenized_dataset = dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

accuracy = evaluate.load("accuracy")


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": accuracy.compute(predictions=predictions, references=labels)}
