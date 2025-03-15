# Environment setup
All packages in the environment is in the ```requirements.txt``` file, though some of them may be redundant for this project. 

We trained the model using the OpenRLHF framework.

# Run the codes
1. Put the data in the ```../data``` folder, such as:
```{python}
data_dir = "../data"

# Define dataset paths
datasets = {
    "VitaminC": f"{data_dir}/VitaminC/test.jsonl",
    "2WikiMultihopQA": f"{data_dir}/2WikiMultihopQA/truncated_first_150.json",
    "Bamboogle": f"{data_dir}/Bamboogle/test.json",
    "FEVER": f"{data_dir}/FEVER/fever_test.jsonl",
    "FEVEROUS": f"{data_dir}/FEVEROUS/feverous_test.jsonl",
    "HotpotQA": f"{data_dir}/HotpotQA/truncated_first_150.json",
    "SVAMP": f"{data_dir}/SVAMP/test.json"
}
```
2. Run ```cot_evaluation.py``` to get the results of COT. Run ```cot_evaluation_refine.py``` to get the results of COT with the refined prompt. Run ```tot_inference.py``` to get the results of TOT.
3. Run ```get_tot_train_data.py``` to get the TOT results for all the training sets.
4. Run ```get_cpo_data.py``` to extract the preference pairs from the TOT results. Run ```get_sft_data.py``` to extract the reasoning path for SFT from the TOT results.
5. Run ```run_cpo_mix_data.sh``` and ```run_sft_mix_data.sh``` to train CPO and SFT respectively.
6. Run ```cpo_evaluation.py``` and ```sft_evaluation.py``` to evaluate CPO and SFT models.
