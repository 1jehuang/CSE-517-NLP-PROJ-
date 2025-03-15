import json
import os

dataset_names = [
    "2WikiMultihopQA",
    "Bamboogle",
    # "FEVER",
    # "FEVEROUS",
    "HotpotQA",
    "SVAMP",
    "VitaminC"
]

all_sft_data = []
for data_name in dataset_names:
    train_tot = json.load(open(f'./results_{data_name}/results_tot_train.json', 'r'))
    sft_data = []
    for i in range(len(train_tot)):
        d = {}
        d['input'] = train_tot[i]['infos'][-1]['x']
        d['output'] = train_tot[i]['infos'][-1]['select_new_ys'][0]

        sft_data.append(d)
    
    # makedir
    os.makedirs(f'./results_{data_name}/sft', exist_ok=True)
    json.dump(sft_data, open(f'./results_{data_name}/sft/train.json', 'w'), indent=4)
    # dump to ./data_download/SFT_training_data_{data_name}.json
    json.dump(sft_data, open(f'./data_download/SFT_training_data_{data_name}.json', 'w'), indent=4)
    
    all_sft_data.extend(sft_data)

# dump to ../data/sft/train.json
os.makedirs(f'../data/sft', exist_ok=True)
json.dump(all_sft_data, open(f'../data/sft/train.json', 'w'), indent=4)
print(len(all_sft_data))
