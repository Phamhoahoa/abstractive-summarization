import yaml
import os
from general_utils import *
import pandas as pd
from transformers import EncoderDecoderModel
import pickle


with open('./config.yaml') as f:
    configs = yaml.load(f, Loader=yaml.SafeLoader)

test_data  = get_data_batch(path='./data/test_tokenized.csv', test = True)
model = EncoderDecoderModel.from_pretrained(configs['output_dir'])
batch_size = configs['batch_size']

def decode_summary(batch):
    # Tokenizer with format: [BOS] <text> [EOS]
    inputs = tokenizer(batch["original"], padding="max_length", truncation=True, max_length=256, return_tensors="pt")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    outputs = model.generate(input_ids, 
                             attention_mask=attention_mask,
                             max_length = configs['decoder_max_length'],
                             early_stopping= configs['early_stopping'],
                             num_beams= configs['num_beams'], 
                             no_repeat_ngram_size= configs['no_repeat_ngram_size'])
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    batch["prediction"] = output_str
    return batch

results = test_data.map(decode_summary, batched=True, batch_size=batch_size, remove_columns=["original"])
rouge_output = rouge.compute(predictions= results["prediction"], references= results["summary"] , rouge_types=["rouge1","rouge2","rougeL"])
file_name = './result-testing/'
os.makedirs(file_name, exist_ok = True)
with open(f'{file_name}decoder.pkl', 'wb') as fs:
    pickle.dump(results["prediction"], fs, protocol=pickle.HIGHEST_PROTOCOL)
with open(f'{file_name}reference.pkl', 'wb') as fs:
    pickle.dump(results["summary"], fs, protocol=pickle.HIGHEST_PROTOCOL)
with open(f'{file_name}result_rouge.txt', 'w+') as fs:
    for indx,val in rouge_output.items():
        fs.write(indx)
        fs.write(' : ')
        fs.write(repr(val.mid))
        f.write('\n')
