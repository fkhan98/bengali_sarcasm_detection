import torch
import pandas as pd
import numpy as np

from model import BertClassifier
from transformers import BertTokenizer

def infer(model, test_df, tokenizer):
    model.eval() 
    ## load the sentences in the test_df i.e test_df['headlines']
    ## write a loop that loops over all the sentences 
    ## inside the loop tokenizes the senteces
    ## the output of the tokenizer contains 'input_ids' and 'attention_mask'
    ## 'input_ids' and 'attention_mask' are needed as input to the model to make inference
    ## see line 58-70 of train.py to understand how input is provided to the model
    ## dont forget to with torch.no_grad() before inference loop
    pass


if __name__ == "__main__":
    
    classifier = BertClassifier()
    classifier.load_state_dict(torch.load('./saved_model/best_model.pt')) ##provide path to model
    tokenizer = BertTokenizer.from_pretrained('sagorsarker/bangla-bert-base') ## tokenizer for BERT
    
    test_df = pd.read_csv('./test.csv') ##provide path to test.csv