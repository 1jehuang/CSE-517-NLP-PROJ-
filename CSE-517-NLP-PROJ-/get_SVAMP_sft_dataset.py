import json

train_tot = json.load(open('./results_SVAMP/results_tot.json', 'r'))

sft_data = []

for i in range(len(train_tot)):
    d = {}
    d['input'] = train_tot[i]['infos'][-1]['x']
    d['output'] = train_tot[i]['infos'][-1]['select_new_ys'][0]

    sft_data.append(d)

json.dump(sft_data, open('./results_SVAMP/sft_data_using_test.json', 'w'), indent=4)    