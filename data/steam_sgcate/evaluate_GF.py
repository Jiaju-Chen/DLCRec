import os
import json
import math
import re
from collections import Counter
input_dir= "./steam_sgcate_result"
path = []
for root, dirs, files in os.walk(input_dir):
    for name in files:
            path.append(os.path.join(input_dir, name))

result_dict = {}
for p in path:
    if 'GF1000gred.json' not in p: continue
    if '' not in p: continue
    print(p)
    f = open(p, 'r')
    test_data = json.load(f)
    f.close()
    text = [_["predict"][0] for _ in test_data] if isinstance(test_data[0]["predict"], list) \
        else [_["predict"] for _ in test_data] 
    target = [_["output"] for _ in test_data] 
    
    predict_genre_list_pattern = re.compile(r'\"\'(.*?)\'') # list
    targetr_genre_pattern = re.compile(r'\[(.*?)\]') # set
    #target_genre_list_pattern = re.compile(r'\[(.*?)\]') # list

    mae = 0
    recall = 0
    ndcg = 0
    recall_list = 0
    control_GP = False
    if 'control_GP' in test_data[0]:
        mae_GP = 0
        recall_GP = 0
        ndcg_GP = 0
        control_GP = True

    for i in range(len(text)):

        genres_predict_all = predict_genre_list_pattern.findall(text[i])
        genres_output_all = predict_genre_list_pattern.findall(test_data[i]['output'])
        #target_genres = targetr_genre_pattern.findall(test_data[i]['instruction'])[0].split('| ')

        if len(genres_predict_all)!=20: 
            print(i, len(genres_predict_all), genres_predict_all)
            continue
        else:
            predicted_genres = genres_predict_all[-10:]
            text_genres = list(set(predicted_genres))
            count = Counter(predicted_genres)
            predicted_genres_dict = dict(count.items())
        
        if len(genres_output_all)!=20: 
            print(i, len(genres_output_all), genres_output_all)
            continue
        else:
            output_genres = genres_output_all[-10:]
            count = Counter(output_genres)
            output_genres_dict = dict(count.items())
            #assert set(output_genres) == set(target_genres)
            target_genres = list(set(output_genres))
        
        if p.split("GF1000")[0][-1] == '_':
            target_cov = len(set(target_genres))
        else:
            # print("controling GPnum")
            target_cov = eval(p.split("GF1000")[0].split('_')[-1])
            
        rec = 0
        dcg = 0
        idcg = 0
        rec_list = 0
        for j in range(target_cov):
            idcg += 1 / math.log2(j + 2)
            if j >= len(text_genres): continue
            if text_genres[j] in target_genres:
                rec += 1
                dcg += 1 / math.log2(j + 2)
        for genre in predicted_genres_dict:
            if genre in output_genres_dict:
                rec_list += min(output_genres_dict[genre], predicted_genres_dict[genre]) / 10
        #print(predicted_genres_dict, output_genres_dict, rec_list)
        ndcg += dcg/idcg
        recall += rec/target_cov
        mae += abs(len(text_genres)- target_cov)

        recall_list += rec_list

        if control_GP:
            GP_instruction = test_data[i]['instruction_controlGP'][test_data[i]['control_GP']]
            GP_genres = targetr_genre_pattern.findall(GP_instruction)[0].split('| ')
            rec_GP = 0
            dcg_GP = 0
            idcg_GP = 0
            rec_list_GP = 0
            for j in range(len(GP_genres)):
                idcg_GP += 1 / math.log2(j + 2)
                if j >= len(text_genres): continue
                if text_genres[j] in GP_genres:
                    rec_GP += 1
                    dcg_GP += 1 / math.log2(j + 2)
                
            ndcg_GP += dcg_GP/idcg_GP
            recall_GP += rec_GP/len(GP_genres)
            mae_GP += abs(len(text_genres)- len(GP_genres))

    mae, recall, ndcg, recall_list = mae/len(text), recall/len(text), ndcg/len(text), recall_list/len(text)
    if control_GP:
        mae_GP, recall_GP, ndcg_GP = mae_GP/len(text), recall_GP/len(text), ndcg_GP/len(text)
    if control_GP:
        result_dict[p] = {"MAE": mae, "RECALL": recall, "NDCG": ndcg, "RECALL_LIST": recall_list, "MAE_GP": mae_GP, "RECALL_GP": recall_GP, "NDCG_GP": ndcg_GP}
    else:
        result_dict[p] = {"MAE": mae, "RECALL": recall, "NDCG": ndcg, "RECALL_LIST": recall_list}

f = open('./steam_sgcate_GF.json', 'w')
json.dump(result_dict, f, indent=4)
