import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import Trainer, TrainingArguments


class JpSentiDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    

def tokenizer_m(train_doc, test_doc):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    model_name = "cl-tohoku/bert-large-japanese"

    id2label = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3, id2label=id2label, label2id=label2id)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_encode = tokenizer(train_doc,return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
    test_encode = tokenizer(test_doc,return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
    return train_encode, test_encode, model, tokenizer

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():

    df_train = pd.read_csv('./output05_train.csv')
    df_test = pd.read_csv('./output05_test.csv')

    test_docs = df_test["Sentence"].tolist()
    test_label = df_test["Emotion"].tolist()
    train_docs = df_train["Sentence"].tolist()
    train_label = df_train["Emotion"].tolist()
    
    train_encodings, test_encodings, model, tokenizer = tokenizer_m(train_docs, test_docs)
    train_dataset = JpSentiDataset(train_encodings, train_label)
    test_dataset = JpSentiDataset(test_encodings, test_label)

    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=40,              # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        save_total_limit=1,              # limit the total amount of checkpoints. Deletes the older checkpoints.
        dataloader_pin_memory=False,  # Whether you want to pin memory in data loaders or not. Will default to True
        evaluation_strategy="steps",
        logging_steps=50,
        logging_dir='./logs'
    )

    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=test_dataset,  # evaluation dataset
        compute_metrics=compute_metrics  # The function that will be used to compute metrics at evaluation
    )

    trainer.train()
    trainer.evaluate(eval_dataset=test_dataset)
    save_directory = "/content/drive/MyDrive/BERT/20230502_sentiment_word_model"
    tokenizer.save_pretrained(save_directory)  
    model.save_pretrained(save_directory)


if __name__ == "__main__":
    main()