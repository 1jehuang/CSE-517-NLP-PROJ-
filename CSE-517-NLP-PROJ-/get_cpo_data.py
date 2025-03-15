import json
from tqdm import tqdm, trange
import os


dataset_names = [
    "2WikiMultihopQA",
    "Bamboogle",
    "FEVER",
    "FEVEROUS",
    "HotpotQA",
    "SVAMP",
    "VitaminC"
]

all_dpo_data = []

for data_name in dataset_names:
    train_tot = json.load(open(f'./results_{data_name}/results_tot_train.json', 'r'))

    dpo_data = []
    # each dpo data contains: x + y, preferred_new_y, dis-preferred_new_y
    # from bottom up, if preferred_thought is None, then the preferred_thought is the last element of select_new_ys
    # if new_y is the prefix of the preferred_thought, then it is the preferred_thought
    # if new_y is not the prefix of the preferred_thought, then it is the dis-preferred_thought
    for tot in tqdm(train_tot):
        # from bottom up, for each layer, get the idx of the preferred_thought
        for i in range(len(tot['infos']) - 1, -1, -1):
            disprefered_list = []
            if i == len(tot['infos']) - 1:
                preferred_thought = tot['infos'][i]['select_new_ys'][0]
                preferred_thought_idx = [0]
            else:
                new_ys = tot['infos'][i]['new_ys']
                values = tot['infos'][i]['values']
                preferred_thought_idx = []
                highest_value = None
                # first traverse, find the highest value
                for j, (new_y, value) in enumerate(zip(new_ys, values)):
                    if preferred_thought.startswith(new_y):
                        highest_value = value
                if highest_value is None:
                    break               
                for j, (new_y, value) in enumerate(zip(new_ys, values)):
                    if preferred_thought.startswith(new_y):
                        preferred_thought_idx.append(j)
                        preferred_thought = new_y
                    elif value < highest_value:
                        disprefered_list.append((new_y, value))
                ys = tot['infos'][i]['ys']
                x = tot['infos'][i]['x']
                # we only care about the parent y that's the prefix of the preferred_thought
                for j in range(len(ys)):
                    if preferred_thought.startswith(ys[j]):
                        # find all the dis-preferred thoughts that starts with ys[j]
                        for disprefered, value in disprefered_list:
                            if disprefered.startswith(ys[j]):
                                # we have a pair here
                                user_content = x + ys[j]
                                # remove the prefix
                                rejected_content = disprefered[len(ys[j]):]
                                chosen_content = preferred_thought[len(ys[j]):]
                                dpo_data.append({
                                    "rejected": [
                                        {
                                            "content": user_content,
                                            "role": "user"
                                        },
                                        {
                                            "content": rejected_content,
                                            "role": "assistant"
                                        }
                                    ],
                                    "rejected_score": value,
                                    "chosen": [
                                        {
                                            "content": user_content,
                                            "role": "user"
                                        },
                                        {
                                            "content": chosen_content,
                                            "role": "assistant"
                                        }
                                    ],
                                    "chosen_score": highest_value
                                })
    # save results
    # makedir
    os.makedirs(f'./results_{data_name}/cpo', exist_ok=True)
    json.dump(dpo_data, open(f'./results_{data_name}/cpo/train.json', 'w'), indent=4)
    
    # dump to ./data_download/CPO_training_data_{data_name}.json
    json.dump(dpo_data, open(f'./data_download/CPO_training_data_{data_name}.json', 'w'), indent=4)

    print(dpo_data[0])
    print(len(dpo_data))
    
    if data_name == "FEVER" or data_name == "FEVEROUS":
        continue
    else:
        all_dpo_data.extend(dpo_data)
    
# save all_dpo_data
## makedir ../data/cpo
os.makedirs(f'../data/cpo_refine', exist_ok=True)
# shuffle the all_dpo_data
import random
random.shuffle(all_dpo_data)
json.dump(all_dpo_data, open('../data/cpo_refine/train.json', 'w'), indent=4)
print(len(all_dpo_data))
    

            