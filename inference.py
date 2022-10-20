import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from sklearn.metrics import f1_score 
from model import BertClassifier, RobertaClassifier
from dataset import Dataset
from transformers import BertTokenizer, RobertaTokenizer

def infer(classifier, test_df, tokenizer):
    headlines = [tokenizer(text, 
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in test_df['headlines']]

    preds = []
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    classifier.eval() 
    
    with torch.no_grad():

        for headline in headlines:
            mask = headline['attention_mask'].to(device)
            input_id = headline['input_ids'].squeeze(1).to(device)
            classifier = classifier.to(device)

            output = classifier(input_id, mask)
            preds.append(output.argmax(dim=1).cpu().numpy()[0])

    test_df['category']=preds
    test_df.drop(columns = ['headlines'], inplace = True)
    test_df.to_csv('test_prediction.csv',index=False)
    


if __name__ == "__main__":
    
    # classifier = BertClassifier()
    # classifier.load_state_dict(torch.load('./saved_model/best_model.pt')) ##provide path to model
    # tokenizer = BertTokenizer.from_pretrained('sagorsarker/bangla-bert-base') ## tokenizer for BERT
    classifier = RobertaClassifier()
    classifier.load_state_dict(torch.load('./saved_model/roberta_model.pt'))
    tokenizer = RobertaTokenizer.from_pretrained('neuralspace-reverie/indic-transformers-bn-roberta')
    
    test_df = pd.read_csv('./test.csv') ##provide path to test.csv
    infer(classifier,test_df,tokenizer)