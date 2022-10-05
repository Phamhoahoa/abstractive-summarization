import glob
import concurrent.futures
from transformers import AutoTokenizer
from transformers import TrainingArguments
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np
from datasets import *
from seq2seq_trainer import Seq2SeqTrainer, UploaderCallback
import yaml
import datasets

with open('./config.yaml') as f:
    configs = yaml.load(f, Loader=yaml.SafeLoader)

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
encoder_max_length=configs['encoder_max_length']
decoder_max_length=configs['decoder_max_length']

def get_dataframe(pathfiles):
    data_df = pd.read_csv(pathfiles)
    return data_df

def process_data_to_model_inputs(batch):                                                               
    # Tokenizer format: [BOS] <text> [EOS]                                               
    inputs = tokenizer(batch["original"], padding="max_length", truncation=True, max_length=encoder_max_length)
    outputs = tokenizer(batch["summary"], padding="max_length", truncation=True, max_length=decoder_max_length)
                                                                                                        
    batch["input_ids"] = inputs.input_ids                                                               
    batch["attention_mask"] = inputs.attention_mask                                                     
    batch["decoder_input_ids"] = outputs.input_ids                                                      
    batch["labels"] = outputs.input_ids.copy()                                                          
    # add padding                                                                             
    batch["labels"] = [                                                                                 
        [-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]
    ]                     
    batch["decoder_attention_mask"] = outputs.attention_mask                                                                              
                                                                                                         
    return batch  

def get_data_batch(path, batch_size=16, test = False ):
    data =  Dataset.from_pandas(df)
    if test:
        return data
    data_batch = data.map(
        process_data_to_model_inputs, 
        batched=True, 
        batch_size=batch_size, 
        remove_columns=["file","original", "summary"],
        )
    data_batch.set_format(
        type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
        )
    
    return data_batch

@dataclass
class Seq2SeqTrainingArguments(TrainingArguments):
    label_smoothing: Optional[float] = field(
        default=0.0, metadata={"help": "The label smoothing epsilon to apply (if not zero)."}
    )
    sortish_sampler: bool = field(default=False, metadata={"help": "Whether to SortishSamler or not."})
    predict_with_generate: bool = field(
        default=False, metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
    )
    adafactor: bool = field(default=False, metadata={"help": "whether to use adafactor"})
    encoder_layerdrop: Optional[float] = field(
        default=None, metadata={"help": "Encoder layer dropout probability. Goes into model.config."}
    )
    decoder_layerdrop: Optional[float] = field(
        default=None, metadata={"help": "Decoder layer dropout probability. Goes into model.config."}
    )
    dropout: Optional[float] = field(default=None, metadata={"help": "Dropout probability. Goes into model.config."})
    attention_dropout: Optional[float] = field(
        default=None, metadata={"help": "Attention dropout probability. Goes into model.config."}
    )
    lr_scheduler: Optional[str] = field(
        default="linear", metadata={"help": f"Which lr scheduler to use."}
    )

# rouge for validate
rouge = datasets.load_metric("rouge")

def compute_metrics(prediction):
    labels_ids = prediction.label_ids
    pred_ids = prediction.predictions

    # remove token khong can thiet
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_res = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

    return {
        "rouge_p": round(rouge_res.precision, 4),
        "rouge_r": round(rouge_res.recall, 4),
        "rouge_f": round(rouge_res.fmeasure, 4),
    }

    
