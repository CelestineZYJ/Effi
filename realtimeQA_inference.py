import argparse 
from typing import Dict, List, Any
import json 
import os 
from tqdm import tqdm 
import bitsandbytes 
from math import ceil 
import re 

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch 
from datasets import load_dataset, Dataset 
from functools import partial 
from dotenv import load_dotenv
from collections import defaultdict 
from UniEval.utils import convert_to_json
from UniEval.metric.evaluator import get_evaluator

load_dotenv() 

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

MODEL_CONFIGS = {
    'llama3': {
        'model_id': "meta-llama/Meta-Llama-3-8B-Instruct",
        'model_path': "/shared/nas2/shared/llms/Meta-Llama-3-8B-Instruct"
    },
    'llama3-text': {
        'model_id': "meta-llama/Meta-Llama-3-8B",
        "model_path": "/shared/nas2/shared/llms/Meta-Llama-3-8B"
    },
    'llama2' : {
        'model_id': 'meta-llama/Llama-2-7b-chat-hf',
        'model_path': "meta-llama/Llama-2-7b-chat-hf"
    },
    "mistral": {
        'model_id': "mistralai/Mistral-7B-Instruct-v0.2",
        "model_path": "/shared/nas2/shared/llms/Mistral-7B-Instruct-v0.2"
    }
}


def construct_context(batch_ins:Dict[str, List], tokenizer, train_strategy, context_type: str='none', prompt='gen', use_template:bool=True) -> Dict[str, List]:
    '''
    
    :use_template: for base models, do not need to use template 
    '''
    new_fields = {
        'text': [],
        'text_len': []
    }
    batch_size = len(batch_ins['question_id']) 
    for idx in range(batch_size):
        question = f"The current time is {batch_ins['question_date'][idx]}." + batch_ins['question_sentence'][idx]

        context = ""
        if context_type == 'gold':
            context = batch_ins['doc'][idx]
        
        if prompt == 'gen':
            if 'SIU' in train_strategy:
                messages = [
                {"role": "user", "content": f"""<s>[INST] <<SYS>>\nYou are a helpful assistant\n<</SYS>>\n\n 
                {question} [/INST] Response: The question is related to the following information, The response to {question} is 
                    """},
                        ]
                # messages = [
                # {"role": "user", "content": f"""<s>[INST] <<SYS>>\nYou are a helpful assistant\n<</SYS>>\n\nYou can check the related document before you answer by putting your checking process enclosed in the <thought> tag. Your answer should be short (a few words or an entity) and specific to the question. Output your final **answer** using the <answer> tag 
                # {question} [/INST] Response: The question is related to the following document, The response to {question} is 
                #     """},
                #         ]
            elif 'Inst' in train_strategy:
                messages = [
                {"role": "user", "content": f"""<s>[INST] <<SYS>>\nYou are a helpful assistant\n<</SYS>>\n\n 
                {question} [/INST] The response to this question is 
                    """},
                        ]
            elif 'index' in train_strategy:
                messages = [
                {"role": "user", "content": f"What company's charcuterie meats have been linked to a multistate salmonella outbreak? "},
                {"role": "assistant", "content": f"The question is related to the document with index charcuterie-recall-salmonella. The answer based on the doucment is Busseto Food"},
                
                {"role": "user", "content": f"What will the NTSB possibly use the found cell phones as? "},
                {"role": "assistant", "content": f"The question is related to the document with index alaska-airlines-plug-door-found-investigation-monday. The answer based on the doucment is Evidence"},
                
                {"role": "user", "content": f"What conference will Washington join next season? "},
                {"role": "assistant", "content": f"The question is related to the document with index 2024-cfp-national-championship-michigan-wolverines-washington-huskies-spt-intl. The answer based on the doucment is Washington will join the Big Ten next season"},
                
                {"role": "user", "content": f"{question}"}
                        ]
            else:
                messages = [
                    {"role": "user", "content": f"""Answer the **question** to the best of your knowledge. Pay attention to the time that the question is asked. You can think before you answer by putting your thought process enclosed in the <thought> tag. Your answer should be short (a few words or an entity). Output your final **answer** using the <answer> tag. 
                    <question> {question} </question>
                        """},
                            ]
            
        elif prompt == 'multichoice':

            option_list = batch_ins['options'][idx]
            option_text = ""
            for opt_idx, opt in enumerate(option_list):
                option_text += f"{opt_idx}. {opt}\n"
            messages = [
                {"role": "user", "content": f"""Answer the **question** to the best of your knowledge. Pay attention to the time that the question is asked. You will be given several options to pick from. You can think before you answer by putting your thought process enclosed in the <thought> tag. Your answer should be the index of the correct option. Output your final **answer** using the <answer> tag. 
                <question> {question} </question>
                <options> {option_text} </options>
                    """},
                        ]
            
        if use_template:
            text = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False)
        else:
            text = messages[0]['content'] + "\n<answer>"

        new_fields['text'].append(text)
        new_fields['text_len'].append(len(tokenizer.tokenize(text))) # text_len is not reliable because of padding 

    return new_fields 

def compute_qa_metrics(gold:List[str], predicted: str):

    gold_answers = [re.sub(r'[^\w\s]','', x) for x in gold]
    gold_answers = [x for x in gold_answers if x!='']
    assert len(gold_answers) >0 
    # Tokenize true and predicted answers
    predicted_answer = re.sub(r'[^\w\s]','',predicted)
    predicted_answer = predicted_answer.strip() 
    pred_tokens = set(predicted_answer.lower().split())
    if len(pred_tokens) == 0:
        return {
        'f1': 0.0, 
        'include_em': 0.0,
        'real_em': 0.0,
        'length': 0.0
        }

    best_f1 = 0.0
    best_length_ratio = 1000

    for true_answer in gold_answers:
        answer_tokens = set(true_answer.lower().split()) 
        # Calculate intersection and union of tokens
        length_ratio = len(pred_tokens) * 1.0 / len(answer_tokens)
        common_tokens = answer_tokens  & pred_tokens
        num_common_tokens = len(common_tokens)
        
        prec = 1.0 * num_common_tokens / len(pred_tokens)
        recall = 1.0 * num_common_tokens / len(answer_tokens )
        # Calculate F1 score
        if num_common_tokens == 0:
            f1_score=0.0
        else:
            f1_score = 2 * (prec * recall) / (prec + recall)
            best_f1 = max(f1_score, best_f1)

        if abs(length_ratio - 1.0) < abs(best_length_ratio - 1.0) :
            best_length_ratio = length_ratio 


    include_em, real_em = 0.0, 0.0
    for true_answer in gold_answers:
        if predicted_answer.lower() == true_answer.lower(): real_em = 1.0
        if  true_answer.lower() in predicted_answer.lower(): include_em = 1.0

    consistency=0.0
    # for true_answer in gold_answers:
    #     data = convert_to_json(output_list=[predicted_answer], src_list=[true_answer.lower()])
    #     # Initialize evaluator for a specific task
    #     evaluator = get_evaluator('fact')
    #     # Get factual consistency scores
    #     consistency = evaluator.evaluate(data, print_result=False)[0]

    return {
        'f1': best_f1, 
        'include_em': include_em,
        'real_em': real_em,
        'consistency': consistency,
        'length': best_length_ratio
    }

if __name__ == '__main__':
    os.environ['HF_TOKEN'] = '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'
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_batch_size', type=int, default=16) 
    parser.add_argument('--prompt', type=str, default='gen')

    parser.add_argument('--model', type=str, choices=['llama3','llama2', 'mistral','llama3-text'], default='llama2')
    args = parser.parse_args() 


    train_strategy = 'indexSumChunkrealtime200' # realtime200  realtime200Inst_indexqa  realtime200SIU_indexqa indexDocrealtime200  realtime200SIU_indexChunkqa  indexChunkrealtime200  indexSumChunkrealtime200 indexDocSamerealtime200
    # index 5e-6   28.5%
    # localpath='/shared/nas2/yujiz/effiUpdating/streamingqa/models/ckpts/shared/nas/data/m1/shared-resource/llm/meta-llama/Llama-2-7b-chat-hf-lr5e-06-seq5000-ratio0.03/final_'+train_strategy+'/epoch_0'
    # index 6.5e-6
    # localpath='/shared/nas2/yujiz/effiUpdating/streamingqa/models/ckpts/shared/nas/data/m1/shared-resource/llm/meta-llama/Llama-2-7b-chat-hf-lr6.5e-06-seq5000-ratio0.03/final_'+train_strategy+'/epoch_0' # realtime200 low include_em
    # pretrain 1e-5
    localpath='/shared/nas2/yujiz/effiUpdating/streamingqa/models/ckpts/shared/nas/data/m1/shared-resource/llm/meta-llama/Llama-2-7b-chat-hf-lr1e-05-seq5000-ratio0.03/final_'+train_strategy+'/epoch_0'
    # siu 1e-6
    # localpath='/shared/nas2/yujiz/effiUpdating/streamingqa/models/ckpts/shared/nas/data/m1/shared-resource/llm/meta-llama/Llama-2-7b-chat-hf-lr1e-06-seq5000-ratio0.03/final_'+train_strategy+'/epoch_1'
    # inst 1e-7
    # localpath='/shared/nas2/yujiz/effiUpdating/streamingqa/models/ckpts/shared/nas/data/m1/shared-resource/llm/meta-llama/Llama-2-7b-chat-hf-lr1e-07-seq5000-ratio0.03/final_'+train_strategy+'/epoch_0'
    # localpath = 'llama2'

    if '50' in train_strategy:
        sample_num = '50'
    elif '200' in train_strategy:
        sample_num = '200'
    elif '400' in train_strategy:
        sample_num = '400'
    elif '600' in train_strategy:
        sample_num = '600'
    elif '780' in train_strategy:
        sample_num = '780'
    dataset = load_dataset("/shared/nas2/yujiz/effiUpdating/streamingqa/data/valrealtime"+sample_num+"Doc_train", token=os.environ['HF_TOKEN'])['train'] # train is the default split here 
    
    if 'lr' in localpath:
        config = AutoConfig.from_pretrained(
        localpath
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            localpath, access_token=os.environ['HF_TOKEN'], padding_side='left'
        )
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
        model = transformers.AutoModelForCausalLM.from_pretrained(
            localpath,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        args.model='ckpt'
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIGS[localpath]['model_id'], access_token=os.environ['HF_TOKEN'], padding_side='left')
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
    
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_CONFIGS[args.model]['model_path'],
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        args.model='zeroshot'


    tokenizer.pad_token = tokenizer.eos_token
     
    tokenized_dataset = dataset.map(partial(construct_context, tokenizer=tokenizer, train_strategy=train_strategy, prompt=args.prompt, context_type='none'), batched=True) 
    tokenized_dataset = tokenized_dataset.map(lambda x: tokenizer(x['text'],padding=True, max_length=2048, truncation=True), 
        batched=True,  
        batch_size=args.gen_batch_size ) # batch size must match for truncation + padding 
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"], output_all_columns=True)
    

    result_list = []
    start = 0
    qa_metrics = defaultdict(list)

    total_batches = ceil(len(tokenized_dataset) /args.gen_batch_size ) 
    for batch_idx in tqdm(range(total_batches)):
        end = min(len(tokenized_dataset), start+ args.gen_batch_size) 
        batch = tokenized_dataset[start: end]
        start += args.gen_batch_size 

        assert(isinstance(batch['input_ids'], torch.Tensor))
        outputs = model.generate(
            input_ids=batch['input_ids'].to(model.device), 
            attention_mask = batch['attention_mask'].to(model.device), 
            max_new_tokens=256,
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id, 
            do_sample=False, # greedy decoding 
            temperature=None,
            top_p=None
        )

        for idx, output_ids in enumerate(outputs):
            input_len = batch['input_ids'][idx].shape[-1]
            result = tokenizer.decode(output_ids[input_len:], skip_special_tokens=True) # str 
            # prompt = tokenizer.decode(batch['input_ids'][idx], skip_special_tokens=True) 

            options = batch['choices'][idx]
            answer_text = [options[int(x)] for x in batch['answer'][idx]]

            m = re.search(r'<answer>([^<]+)', result)
            if m:
                predicted = m.group(1)

            else:
                predicted = result 

            results = {
                'question_id': batch['question_id'][idx],
                'question_date': batch['question_date'][idx],
                'question_sentence': batch['question_sentence'][idx],
                'answer_text': answer_text,
                'predicted': predicted, 
                'generated': result 
            }
            try:
                metrics = compute_qa_metrics(answer_text, predicted)
                for metric in metrics:
                    qa_metrics[metric].append(metrics[metric]) 
                if metrics['include_em'] ==1.0:
                    print(results['question_id'])


                results.update(metrics)
            except AssertionError:
                print(f"question {batch['question_id'][idx]} has no gold answer")
                continue 

            result_list.append(results)

    

    filename = f'output/realtimeQA_llama2/{train_strategy}_{args.model}_{args.prompt}.json' 


    with open(filename,'w') as f:
        json.dump(result_list, f, indent=2)


    final_metrics = {} 
    for metric in qa_metrics:
        final_metrics[metric] = sum(qa_metrics[metric]) *1.0 / len(qa_metrics[metric]) 

    

    with open(f'output/realtimeQA_llama2/{train_strategy}_{args.model}_{args.prompt}_metrics.json','w') as f:
        json.dump(final_metrics, f, indent=2) 


