from transformers import GenerationConfig, AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import os
import math
import json
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

base_model = "../Meta-Llama-3-8B-Instruct/"
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto",
)

f = open('./data/steam_sgcate/steam_over20.dat', 'r', encoding='ISO-8859-1')
movies = f.readlines()
movie_names = [_.split(' %% ')[1] for _ in movies]
movie_ids = [_ for _ in range(len(movie_names))]
movie_dict = dict(zip(movie_names, movie_ids))
result_dict = dict()

model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2
model.half()  # seems to fix bugs for some users.
#model.eval()
tokenizer.padding_side = "left"


def batch(list, batch_size=1):
    chunk_size = (len(list) - 1) // batch_size + 1
    for i in range(chunk_size):
        yield list[batch_size * i: batch_size * (i + 1)]

       
movie_embeddings = []
from tqdm import tqdm
with torch.no_grad():
    for i, batch_input in tqdm(enumerate(batch(movie_names, 4))):
        input = tokenizer(batch_input, return_tensors="pt", padding=True).to(model.device)
        input_ids = input.input_ids
        attention_mask = input.attention_mask
        outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        movie_embeddings.append(hidden_states[-1][:, -1, :].detach().cpu())

movie_embeddings = torch.cat(movie_embeddings, dim=0)
torch.save(movie_embeddings, "./data/steam_sgcate/steam_embedding_llama3.pt")
