import os
import json
import math
import re

input_dir= "./steam_sgcate_result"
path = []
for root, dirs, files in os.walk(input_dir):
    for name in files:
            path.append(os.path.join(input_dir, name))
print(path)

result_dict = {}
for p in path:
    if 'GP1000gred' not in p: continue
    print(p)
    f = open(p, 'r')
    test_data = json.load(f)
    f.close()
    text = [_["predict"][0] for _ in test_data] if isinstance(test_data[0]["predict"], list) \
        else [_["predict"] for _ in test_data] 
    target = [_["output"] for _ in test_data] 

    mae = 0
    recall = 0
    ndcg = 0
    for i in range(len(text)):
        text_items = text[i].split('| ')
        target_items = target[i].split('| ')
        
        
        match = re.search(r'the (\d+) genres that the user is most likely to enjoy in the future', test_data[i]['instruction'])
        
            
        if match:
            number = int(match.group(1))  
            target_cov = number
        else:
            match = re.search(r'the (\d+) most likely genres', test_data[i]['instruction'])
            if match: 
                number = int(match.group(1))  
                target_cov = number
            else:
                print('No match found.')
                target_cov = len(target_items)

        rec = 0
        dcg = 0
        idcg = 0
        for j in range(len(target_items)):
            idcg += 1 / math.log2(j + 2)
            if j >= len(text_items): continue
            if text_items[j] in target_items:
                rec += 1
                dcg += 1 / math.log2(j + 2)
            
        ndcg += dcg/idcg
        recall += rec/len(target_items)
        mae += abs(len(text_items)-target_cov)
    mae, recall, ndcg = mae/len(text), recall/len(text), ndcg/len(text)
    result_dict[p] = {"MAE": mae, "RECALL": recall, "NDCG": ndcg}

f = open('./steam_sgcate_GP.json', 'w')    
json.dump(result_dict, f, indent=4)