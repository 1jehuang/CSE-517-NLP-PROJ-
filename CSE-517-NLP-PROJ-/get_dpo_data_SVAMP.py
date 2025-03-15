import json
from tqdm import tqdm, trange


train_tot = json.load(open('./results_SVAMP/results_tot.json', 'r'))

dpo_data = []
# each dpo data contains: x + y, preferred_new_y, dis-preferred_new_y
# from bottom up, if preferred_thought is None, then the preferred_thought is the last element of select_new_ys
# if new_y is the prefix of the preferred_thought, then it is the preferred_thought
# if new_y is not the prefix of the preferred_thought, then it is the dis-preferred_thought
for tot in tqdm(train_tot):
    # from bottom up, for each layer, get the idx of the preferred_thought
    disprefered_list = []
    for i in range(len(tot['infos']) - 1, -1, -1):
        if i == len(tot['infos']) - 1:
            preferred_thought = tot['infos'][i]['select_new_ys'][0]
            preferred_thought_idx = [0]
        else:
            new_ys = tot['infos'][i]['new_ys']
            preferred_thought_idx = []
            for j, new_y in enumerate(new_ys):
                if preferred_thought.startswith(new_y):
                    preferred_thought_idx.append(j)
                    preferred_thought = new_y
                else:
                    disprefered_list.append(new_y)
            ys = tot['infos'][i]['ys']
            x = tot['infos'][i]['x']
            # we only care about the parent y that's the prefix of the preferred_thought
            for j in range(len(ys)):
                if preferred_thought.startswith(ys[j]):
                    # find all the dis-preferred thoughts that starts with ys[j]
                    for disprefered in disprefered_list:
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
                                "rejected_score": 0.0,
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
                                "chosen_score": 1.0
                            
                            })


print(dpo_data[0])
print(len(dpo_data))
json.dump(dpo_data, open('./results_SVAMP/dpo_data_using_test.json', 'w'), indent=4)
                            
            
            
            
            