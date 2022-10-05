import glob, random
import pandas as pd
import concurrent.futures
import numpy as np
train_paths = glob.glob('./data/train_tokenized/*')
val_paths = glob.glob('./data/val_tokenized/*')
test_paths = glob.glob('./data/test_tokenized/*')

def read_content(filename):
    with open(filename) as f:
        rows  = f.readlines()
        original = ' '.join(''.join(rows[4:]).split('\n'))
        summary = ' '.join(rows[2].split('\n'))
            
    return {'file' : filename,'original': original, 'summary': summary}
def get_dataframe(filenames):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        data = executor.map(read_content, filenames)
    data_df = pd.DataFrame(data)
    data_df.dropna(inplace = True)
    data_df = data_df.sample(frac=1).reset_index(drop=True)

    return data_df
  
train_df = get_dataframe(train_paths)
train_df.to_csv('./train_tokenized.csv', index = False)
val_df =  get_dataframe(val_paths)
val_df.to_csv('./val_tokenized.csv', index = False)
test_df =  get_dataframe(test_paths)
test_df.to_csv('./test_tokenized.csv', index = False)