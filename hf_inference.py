import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["WORLD_SIZE"] = "1"
os.environ["CACHE_PATH"] = "/shared/nas2/yujiz/effiUpdating/llmCache/"
import numpy as np
import pandas as pd
from typing import Dict, List, Mapping, Union
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig


import datasets
import json

from accelerate import Accelerator, DistributedType
import accelerate
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from collections import defaultdict
import time

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    MistralForCausalLM,
    default_data_collator,
    get_scheduler,
)
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
import warnings
import torch
import gzip
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import f1_score
from tqdm import tqdm

import random

# Set the CUDA device

device='cuda'

def compute_f1_score(true_answers, predicted_answer):
    """Compute the F1 score for a QA task."""
    # Tokenize true and predicted answers
    true_tokens = set()
    for true_answer in true_answers:
        true_tokens.update(set(true_answer.lower().split()))
    
    pred_tokens = set(predicted_answer.lower().split())
    
    # Calculate intersection and union of tokens
    common_tokens = true_tokens.intersection(pred_tokens)
    num_common_tokens = len(common_tokens)
    
    if num_common_tokens == 0:
        return 0.0
    
    precision = num_common_tokens / len(pred_tokens)
    recall = num_common_tokens / len(true_tokens)
    
    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    return f1_score

# Function to generate answer for a given question
def generate_answer(model, tokenizer, device, question, max_length=100):
    # Tokenize the question
    input_ids = tokenizer(question, return_tensors="pt").input_ids.to(device)
    # Generate answer using LLAMA causal LM
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    # Decode the generated output
    generated_answer = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return generated_answer

def main():
    accelerator = Accelerator()
    accelerator.wait_for_everyone()
    accelerator.print("Start loading dataset")
    
    raw_datasets = load_dataset("/shared/nas2/yujiz/effiUpdating/streamingqa/data")
    
    localpath = "/shared/nas2/yujiz/effiUpdating/streamingqa/models/ckpts/TinyLlama/TinyLlama-1.1B-Chat-v1.0-lr3e-5-seq1024-ratio0.03/final"
    # localpath = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" zero-shot inference
    accelerator.print("Finish loading dataset")
    config = AutoConfig.from_pretrained(
        localpath
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        localpath
    )
    accelerator.print('Finish loading config')
    accelerator.print("Start loading pretrained")
        
    model = transformers.AutoModelForCausalLM.from_pretrained(
        localpath,
        config=config
    )
    accelerator.print("Finish loading pretrained")
    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        # model.resize_token_embeddings(len(tokenizer))
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
        accelerator.print(f"Embedding size updated to {model.get_input_embeddings().weight.shape[0]}")
    accelerator.wait_for_everyone()
    
 
    # Define the file paths for the QA dataset
    _file_name_by_streamingqa_subset = {
        'train': '/shared/nas2/yujiz/effiUpdating/streamingqa/downloadData/streaminqa_train.jsonl.gz',
        'valid': '/shared/nas2/yujiz/effiUpdating/streamingqa/downloadData/streaminqa_valid.jsonl.gz',
        'eval': '/shared/nas2/yujiz/effiUpdating/streamingqa/downloadData/streaminqa_eval.jsonl.gz',
    }

    # Load the QA dataset
    streamingqa_eval = []
    with gzip.open(_file_name_by_streamingqa_subset['eval'], 'rb') as input_file:
        for line in input_file:
            streamingqa_eval.append(json.loads(line.decode()))
    
    # Perform zero-shot inference and calculate F1 score
    true_answers = []
    predicted_answers = []
    f1_scores = []
    print(len(streamingqa_eval))
    model = model.to(device)
    streamingqa_eval = random.sample(streamingqa_eval, 1000)
    for qa in tqdm(streamingqa_eval):
        question = qa['question']
        true_answer = qa['answers']  # Assuming the first answer is the ground truth
        
        generated_answer = generate_answer(model, tokenizer, device, question)
        # print('*'*100)
        # print(generated_answer)
        # print('$'*100)
        # print(true_answer)

        # Collect true and predicted answers
        f1 = compute_f1_score(true_answer, generated_answer)
        f1_scores.append(f1)
        # print(f1)


    avg_f1_score = sum(f1_scores) / len(f1_scores)
    print("Average F1 Score:", avg_f1_score)
        
if __name__ == "__main__":
    main()
