import json
import re
from tqdm import tqdm

# modify the instrucition to make use of the control of GP

GP_paths = [
    # "./steam_sgcate_result/GP_ep25_1_-1_5e-4_0.05_2GP1000gred.json",
    # "./steam_sgcate_result/GP_ep25_1_-1_5e-4_0.05_5GP1000gred.json",
    # "./steam_sgcate_result/GP_ep25_1_-1_5e-4_0.05_8GP1000gred.json",
    # "./steam_sgcate_result/GP_ep25_1_-1_5e-4_0.05_GP1000gred.json",
]

GF_test_data_path = './data/steam_sgcate/test_1000_BERT_GF.json'
GF_test_data_path_controlGP = './data/steam_sgcate/test_1000_BERT_GF_controlGP.json'

GP_predictions = {}
target_genere_pattern = r'\[(.*?)\]'

for p in GP_paths:
    with open(p, 'r') as f:
        test_data_GP = json.load(f)
        predict = [_['predict'] for _ in test_data_GP]
        GP_predictions[p] = predict

with open(GF_test_data_path, 'r') as f:
    test_data = json.load(f)
    for i, test in enumerate(test_data):
        test_data[i]['instruction_controlGP'] = {}
        for p in GP_paths:
            GP_genres = GP_predictions[p][i]
            replaced_string = re.sub(target_genere_pattern, lambda match: f"[{GP_genres[0]}]", test_data[i]['instruction'])
            key_p = p.split('/')[-1][:-5]
            test_data[i]['instruction_controlGP'][key_p] = replaced_string
            
with open(GF_test_data_path_controlGP, 'w') as f:
    json.dump(test_data, f, indent=4)
