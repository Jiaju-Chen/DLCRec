from transformers import GenerationConfig, AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import os
import math
import json
import re
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
parse = argparse.ArgumentParser()
parse.add_argument("--input_dir",type=str, default="./movie_sgcate_result", help="your model directory")

args = parse.parse_args()

path = []
for root, dirs, files in os.walk(args.input_dir):
    for name in files:
            path.append(os.path.join(args.input_dir, name))

base_model = "../Meta-Llama-3-8B-Instruct/"
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto",
)

def div_cov(id_list, movie_category_dict):
    previous_category = set()
    div_list = []
    for i in id_list:
        i_category = set(movie_category_dict[i])
        union = previous_category.union(i_category)
        intersection = previous_category.intersection(i_category)
        jaccard_similarity = len(intersection) / len(union)
        div_list.append((1 - jaccard_similarity)/len(id_list))
        previous_category = union
    return sum(div_list), len(previous_category)

def cal_match(id_list, genre_list, movie_category_dict):
    cnt_match = 0
    for id, genre in zip(id_list, genre_list):
        if movie_category_dict[id][0] == genre:
            cnt_match += 1

    return cnt_match / len(id_list)

f = open('./data/movie/movies.dat', 'r', encoding='ISO-8859-1')
movies = f.readlines()
movie_names = [_.split('::')[1].strip("\"") for _ in movies]
movie_ids = [_ for _ in range(len(movie_names))]
movie_dict = dict(zip(movie_names, movie_ids))
movie_categories_multi = [_.split('::')[2].rstrip("\n").split('|') for _ in movies]
movie_categories =  [[_[0]] for _ in movie_categories_multi]
movie_category_dict = dict(zip(movie_ids, movie_categories))
result_dict = dict()

predict_genre_pattern = re.compile(r'\"\'(.*?)\'')
predict_item_pattern = re.compile(r'\"(.*?)\"\'')

for p in path:
    if 'IP1000gred' not in p: continue
    print(p)
    result_dict[p] = {
        "NDCG": [],
        "HR": [],
        "REP_RATE": [],
        "DIV": []
    }
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    model.eval()
    f = open(p, 'r')
    import json
    test_data = json.load(f)
    f.close()
    text = [_["predict"][0] for _ in test_data] if isinstance(test_data[0]["predict"], list) else [_["predict"] for _ in test_data] 
    tokenizer.padding_side = "left"

    def batch(list, batch_size=1):
        chunk_size = (len(list) - 1) // batch_size + 1
        for i in range(chunk_size):
            yield list[batch_size * i: batch_size * (i + 1)]
    
    # print(text[0])

    predict_embeddings = []
    from tqdm import tqdm
    for i, batch_input in tqdm(enumerate(text)):
        all_items = predict_item_pattern.findall(batch_input)
        if len(all_items) != 20: 
            print(i, len(all_items), all_items)
            items = all_items[10:20]
            items = items + ['NaN'] * (10-len(items)) 
        else:
            items = all_items[10:20]
        for j, item in enumerate(items):
            if item in test_data[i]['input']:
                print(f"{i}th sample's {j}th item {item} happens in history")
                items[j] = 'NaN' # rank-> 999
        #batch_input.replace("\"", "").split('| ')[:10]
        #print(items)

        input = tokenizer(items, return_tensors="pt", padding=True).to("cuda")
        input_ids = input.input_ids
        attention_mask = input.attention_mask
        outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        predict_embeddings.append(hidden_states[-1][:, -1, :].detach().cpu())
    
    predict_embeddings = torch.cat(predict_embeddings, dim=0).cuda()
    print(f"Predict_embeddings' shape is {predict_embeddings.shape}.")
    movie_embedding = torch.load("./data/movie/movie_embedding_llama3.pt").cuda()
    dist = torch.cdist(predict_embeddings, movie_embedding, p=2)
        
    rank = dist
    # rank = rank.argsort(dim = -1).argsort(dim = -1)
    # topk_values, topk_indices = torch.topk(dist, 1, dim=-1, largest=False)
    min_value, min_indices = torch.min(dist, dim=-1)

    topk_list = [10]
    NDCG = []
    HR = []
    REP_RATE = []
    DIV = []
    COV  = []
    COV_MAE = [] # coverage MAE between the output genres and the target genres.
    MATCH_T = [] # matching rate of target movies
    MATCH_P = [] # matching rate of predictions
    for topk in topk_list:
        sum_ndcg = 0   
        sum_hr = 0
        sum_rep_rate = 0
        sum_div = 0
        sum_cov = 0
        sum_cov_mae = 0
        sum_match_t = 0
        sum_match_p = 0
        for i in range(len(test_data)):
            all_items_target = predict_item_pattern.findall(test_data[i]['output'])
            target_movies = all_items_target[10:20]
            if len(all_items_target)!= 20:
                print("length error in target_movies!", i, len(all_items_target), all_items_target)

            all_genres_target = predict_genre_pattern.findall(test_data[i]['output'])
            target_genres = all_genres_target[10:20]
            if len(all_genres_target)!= 20:
                print("length error in target_genres!", i, len(all_genres_target), all_genres_target)
            
            if "GF1000" not in p:
                target_cov = len(set(target_genres))
            elif p.split("GF1000")[0][-1] == '_':
                target_cov = len(set(target_genres))
            else:
                print("controling GPnum")
                target_cov = eval(p.split("GF1000")[0].split('_')[-1])

            all_genres_predict = predict_genre_pattern.findall(test_data[i]['predict'][0])
            predict_genres = all_genres_predict[10:20]
            if len(all_genres_predict)!= 20:
                print("length error in predict_genres!", i, len(all_genres_predict), all_genres_predict)
            
            target_movie_ids = [movie_dict[target_movie] for target_movie in target_movies]
            #rankIds = [rank[j+10*i][target_movie_id].item() for j, target_movie_id in zip(range(10), target_movie_ids)]
            rec_id_list = min_indices[10*i:10*i+topk].tolist()
            div, cov = div_cov(rec_id_list, movie_category_dict)
            cov_mae = abs(cov - target_cov)
            match_t = cal_match(rec_id_list, target_genres, movie_category_dict)
            match_p = cal_match(rec_id_list, predict_genres, movie_category_dict)
            ndcg, hr = 0, 0
            for j, rec_id in enumerate(rec_id_list[:topk]):
                if rec_id in target_movie_ids[:topk]:
                    ndcg += (1 / math.log(j + 2))
                    hr += 1
            idcg = 0
            for j in range(topk):
                idcg += (1 / math.log(j + 2))
            sum_ndcg += ndcg / idcg
            sum_hr += hr / topk
            sum_rep_rate += 1 - len(set(rec_id_list)) / topk
            sum_div += div
            sum_cov += cov
            sum_cov_mae += cov_mae
            sum_match_t += match_t
            sum_match_p += match_p
        NDCG.append(sum_ndcg / len(test_data))
        HR.append(sum_hr / len(test_data))
        REP_RATE.append(sum_rep_rate / len(test_data))
        DIV.append(sum_div  / len(test_data))
        COV.append(sum_cov  / len(test_data))
        COV_MAE.append(sum_cov_mae  / len(test_data))
        MATCH_T.append(sum_match_t / len(test_data))
        MATCH_P.append(sum_match_p / len(test_data))

    print(NDCG)
    print(HR)
    print(REP_RATE)
    print(DIV)
    print(COV)
    print(COV_MAE)
    print(MATCH_T)
    print(MATCH_P)
    print('_' * 100)
    result_dict[p]["NDCG"] = NDCG
    result_dict[p]["HR"] = HR
    result_dict[p]["REP_RATE"] = REP_RATE
    result_dict[p]["DIV"] = DIV
    result_dict[p]["COV"] = COV
    result_dict[p]["COV_MAE"] = COV_MAE
    result_dict[p]["MATCH_T"] = MATCH_T
    result_dict[p]["MATCH_P"] = MATCH_P

    
f = open('./result_IP_movie.json', 'w')
json.dump(result_dict, f, indent=4)
