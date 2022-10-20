## write a test script on the valid.csv to measure the f1 score 
## A helper function to calculate f1 score needs to be written
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score 
from model import BertClassifier
from dataset import Dataset
from transformers import BertTokenizer



def valid_test(model, test_df, tokenizer):
    
    val_data = Dataset(test_df)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model.eval() 
    preds = []
    with torch.no_grad():

        for headline in val_data.headline:
            mask = headline['attention_mask'].to(device)
            input_id = headline['input_ids'].squeeze(1).to(device)

            output = classifier(input_id, mask)
            preds.append(output.argmax(dim=1).cpu().numpy()[0])
    
    f1_score_output = f1_score(val_data.labels, preds)
    return f1_score_output


if __name__ == "__main__":
    
    classifier = BertClassifier()
    classifier.load_state_dict(torch.load('./saved_model/best_model.pt')) ##provide path to model
    tokenizer = BertTokenizer.from_pretrained('sagorsarker/bangla-bert-base') ## tokenizer for BERT
    
    df_val = pd.read_csv('./val.csv')  ##provide path to test.csv
    valid_test(classifier,df_val,tokenizer)