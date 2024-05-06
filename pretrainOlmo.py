import os
from datasets import load_dataset
# from transformers import pipeline
# import hf_olmo # pip install ai2-olmo
from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.environ['CACHE_DIR'] = '/shared/nas2/yujiz/cache_storage'
os.environ["HF_DATASETS_CACHE"] = '/shared/nas2/yujiz/cache_storage'

os.environ["DATA_DIR"] = "/shared/nas2/yujiz/llm_entropy/dolmaData"
dataset = load_dataset("dolma", split="train")
print('have loaded dolma dataset')


# olmo = AutoModelForCausalLM.from_pretrained("allenai/OLMo-7B-Instruct", trust_remote_code=True)
# olmo = olmo.to('cuda')
# tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-7B-Instruct")
# samples = ['Language modeling is ', 'Who is Mary Lee Pfeiffer\'s son', 'Who is Tom Cruise\'s mother', 'tell me some outstanding female ai scientists']
# samples = [ 'tell me some outstanding female ai scientists']
# for message in samples:
#     message='tell me some outstanding female ai scientists'
#     inputs = tokenizer(message, return_tensors='pt', return_token_type_ids=False)
#     # optional verifying cuda
#     inputs = {k: v.to('cuda') for k,v in inputs.items()}
#     response = olmo.generate(**inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
#     print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])



from transformers import AutoModelForCausalLM, AutoTokenizer
olmo = AutoModelForCausalLM.from_pretrained("allenai/OLMo-7B-Instruct", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-7B-Instruct")
chats = [
    [
        { "role": "user", "content": "tell me some outstanding female ai scientists" }
        ],
    [{ "role": "user", "content": "Who is Tom Cruise\'s mother" }],
    [{ "role": "user", "content": "Who is Mary Lee Pfeiffer\'s son" }],
    [{ "role": "user", "content": "Who is not an outstanding scientist?" }]
]
for chat in chats:
    # chat = [
    # { "role": "user", "content": "tell me some outstanding female ai scientists" },
    # ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    # optional verifying cuda
    # inputs = {k: v.to('cuda') for k,v in inputs}
    olmo = olmo.to('cuda')
    response = olmo.generate(input_ids=inputs.to(olmo.device), max_new_tokens=2000, do_sample=True, top_k=50, top_p=0.95)
    print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])
