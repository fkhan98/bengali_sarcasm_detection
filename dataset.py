import torch
import pandas as pd
import numpy as np

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('sagorsarker/bangla-bert-base')

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.labels = [label for label in df['category']]
        self.headline = [tokenizer(text, 
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in df['headlines']]
        
    def __len__(self):
        return len(self.labels)
    
    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])
    
    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.headline[idx]
    
    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        
        return batch_texts, batch_y
        
if __name__ == "__main__":
    df_train = pd.read_csv('./train.csv') 
    df_val = pd.read_csv('./val.csv') 
    demo = Dataset(df_train)