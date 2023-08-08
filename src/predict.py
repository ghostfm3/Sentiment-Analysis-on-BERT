import torch
import matplotlib as plt
import sys
import matplotlib.pyplot as plt
import japanize_matplotlib
import ginza
import spacy
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

def ja_ginza_token(path):
    word_list = []

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            word_list.append(line)

    nlp = spacy.load('ja_ginza')
    ginza.set_split_mode(nlp, 'C')
    doc = nlp(''.join(word_list))
    pos_list = ["NOUN"]
    tokens = [token.text for token in doc if token.pos_ in pos_list]
    return tokens

def sentiment__analysis(words):
    result_dict = {}
    save_directory = "./20230415_sentiment_word_model"
    tokenizer = AutoTokenizer.from_pretrained(save_directory)
    model = AutoModelForSequenceClassification.from_pretrained(save_directory)
    sentiment_analyzer = pipeline("sentiment-analysis", model=model.to("cpu"), tokenizer=tokenizer)
    p_count=0
    n_count=0
    ne_count=0

    for token in words:
        result = sentiment_analyzer(token)
        result_dict[token] = result[0]['label']
        if result[0]['label'] == 'POSITIVE':
            p_count += 1
        elif result[0]['label'] == 'NEGATIVE':
            ne_count += 1
        elif result[0]['label'] == 'NEUTRAL':
            n_count += 1
    print(f"positive:{p_count}, negative:{ne_count}, neutral:{n_count}")
    return p_count, ne_count, n_count, result_dict

def drow_graph(count1, count2, count3):
    labels = ['Positive','Neutral' ,'Negative']
    sizes = [count1, count2, count3]
    colors = ['blue','gray' ,'red']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Emotional rate')
    plt.savefig('./result.png')

def main():
    file_path = sys.argv[1]
    word_list = ja_ginza_token(file_path)
    positive, negative, neutral, r_dict = sentiment__analysis(word_list)
    drow_graph(positive, neutral, negative)

    with open('output.json', 'w') as f:
        json.dump(r_dict, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
