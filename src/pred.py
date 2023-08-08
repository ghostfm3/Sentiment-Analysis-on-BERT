import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import sys

def main():
    save_directory = "/content/drive/MyDrive/BERT/20230415_sentiment_word_model"
    tokenizer = AutoTokenizer.from_pretrained(save_directory)
    model = AutoModelForSequenceClassification.from_pretrained(save_directory)
    sentiment_analyzer = pipeline("sentiment-analysis", model=model.to("cpu"), tokenizer=tokenizer)
    text = sys.argv[1]
    result = sentiment_analyzer(text)
    print(f"判定：{result[0]['label']}")

if __name__ == "__main__":
    main()