## write a test script on the valid.csv to measure the f1 score 
## A helper function to calculate f1 score needs to be written
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from sklearn.metrics import f1_score 
from model import BertClassifier, RobertaClassifier
from dataset import Dataset
from transformers import BertTokenizer, RobertaTokenizer



def valid_test(model, test_df, tokenizer):
    
    val_data = Dataset(test_df, tokenizer)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)
    
    model.eval() 
    preds = []
    with torch.no_grad():

        for headline in val_data.headline:
            mask = headline['attention_mask'].to(device)
            input_id = headline['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)
            preds.append(output.argmax(dim=1).cpu().numpy()[0])
    
    f1_score_output = f1_score(val_data.labels, preds)
    return f1_score_output


if __name__ == "__main__":
    
    classifier = BertClassifier()
    classifier.load_state_dict(torch.load('/home/fahim/codes/ETE_Comp/saved_model/best_model.pt')) ##provide path to model
    tokenizer = BertTokenizer.from_pretrained('sagorsarker/bangla-bert-base') ## tokenizer for BERT
    # roberta_classifier = RobertaClassifier()
    # roberta_classifier.load_state_dict(torch.load('./saved_model/roberta_model.pt')) 
    # tokenizer = RobertaTokenizer.from_pretrained('neuralspace-reverie/indic-transformers-bn-roberta') ## tokenizer for ROBERTA
    
    df_test = pd.read_csv('./train.csv')  ##provide path to test.csv
    print(valid_test(classifier,df_test,tokenizer))