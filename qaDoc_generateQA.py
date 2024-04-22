import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
train_strategy = 'onlineAdapt'
# localpath = "/shared/nas2/yujiz/effiUpdating/streamingqa/models/ckpts/NousResearch/Llama-2-7b-chat-hf-lr6.5e-06-seq1600-ratio0.03/final_"+train_strategy
localpath = "NousResearch/Llama-2-7b-chat-hf" # zero-shot inference/
# localpath = "NousResearch/Meta-Llama-3-8B-Instruct"
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
def generate_answer(model, tokenizer, device, question, max_length=1500):
    # Tokenize the question
    input_ids = tokenizer(question, return_tensors="pt").input_ids.to(device)
    # Generate answer using LLAMA causal LM
    output = model.generate(input_ids, max_length=max_length,temperature=0.3,top_k=50,top_p=0.95,num_return_sequences=1)
    # Decode the generated output
    generated_answer = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return generated_answer

def read_qa_list_from_json(file_path):
        qa_list = []
        with open(file_path, 'r') as f:
            for line in f:
                qa_list.append(json.loads(line))
        return qa_list

def extract_qa_pairs(generated_QA_answers, article):
    qa_pairs = []
    lines = generated_QA_answers.strip().split('\n')
    for line in lines:
        if line.startswith('Q'):
            q_value = line.split(':')[1].strip()
        elif line.startswith('A'):
            a_value = line.split(':')[1].strip()
            qa_pairs.append({'context':article,'question': q_value, 'answer': a_value})
    return qa_pairs

def main():
    accelerator = Accelerator()
    accelerator.wait_for_everyone()
    
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

    # self information update generates question-answer pairs
    true_answers = []
    predicted_answers = []
    f1_scores = []
    streamingqa_eval = read_qa_list_from_json('/shared/nas2/yujiz/effiUpdating/streamingqa/data/'+train_strategy+'Doc_train.json')
    print(len(streamingqa_eval))
    model = model.to(device)
    siu_train = []
    for qa in tqdm(streamingqa_eval):
        question = qa['question']
        true_answer = qa['answer']  # Assuming the first answer is the ground truth
        
        # question = 'You are a helpful assistent. Please answer the question with only one sentence as the answer. The question is: '+question
        question="You are a helpful assistant. You generate question-answer pairs from a given news article to help extract infromation from this article. If the answer is not given in this article, skip this question. You generate the question-answer pairs in the format of Q1 A1 Q2 A2 Q3 A3 Q4 A4.\n "+"The article is: "+qa['context']
        # question="You are a helpful assistant. You raise questions from a given news article to help extract infromation from this article. Please generate high-quality question-answer pairs based on this article.\n"+"The article is: "+qa['context']       
        generated_answer = generate_answer(model, tokenizer, device, question)
        print('*'*100)
        print(generated_answer) #[len(question):]
        siu_train.extend(extract_qa_pairs(generated_QA_answers=generate_answer, article=qa['context']))
        # print('$'*100)
        # print(true_answer)
        x=1

    # # Perform zero-shot inference and calculate F1 score
    # true_answers = []
    # predicted_answers = []
    # f1_scores = []
    # streamingqa_eval = read_qa_list_from_json('/shared/nas2/yujiz/effiUpdating/streamingqa/data/'+train_strategy+'Doc_train.json')
    # print(len(streamingqa_eval))
    # model = model.to(device)
    # for qa in tqdm(streamingqa_eval):
    #     question = qa['question']
    #     true_answer = qa['answer']  # Assuming the first answer is the ground truth
    #     if true_answer == 'Dominic Perrottet':
    #         x=1
    #     question = 'You are a helpful assistent answering the given question. Please only output one sentence as the answer. The question is: '+question
    #     generated_answer = generate_answer(model, tokenizer, device, question)
    #     generated_answer = generated_answer.replace('The answer is:','')
    #     generated_answer = generated_answer.replace('Answer:','')
    #     generated_answer = generated_answer.replace('answer:','')
    #     print('*'*100)
    #     print(generated_answer) #[len(question):]
    #     print('$'*100)
    #     print(true_answer)

    #     # Collect true and predicted answers
    #     f1 = compute_f1_score([true_answer], generated_answer[len(question):])
    #     f1_scores.append(f1)
    #     print(f1)
    #     avg_f1_score = sum(f1_scores) / len(f1_scores)
    #     print("Average F1 Score:", avg_f1_score)
    #     x=1

    # print(f1_scores)
    # avg_f1_score = sum(f1_scores) / len(f1_scores)
    # print("Average F1 Score:", avg_f1_score)
        
if __name__ == "__main__":
    main()
