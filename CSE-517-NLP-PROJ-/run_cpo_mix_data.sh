set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_dpo \
   --save_path ./checkpoint/llama3-1b-cpo-1epoch-mix_data-refine \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 16 \
   --micro_train_batch_size 2 \
   --pretrain meta-llama/Llama-3.2-1B-Instruct \
   --bf16 \
   --max_epochs 1 \
   --max_len 2048 \
   --zero_stage 3 \
   --learning_rate 2e-6 \
   --beta 0.1 \
   --dataset json@/home/NETID/zengsh/course/data/cpo_refine \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --flash_attn \
   --gradient_checkpointing \
   --max_ckpt_num 2  
EOF
    # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
    # --ipo [for IPO]
    # --label_smoothing 0.1 [for cDPO]
    # --ref_offload
    # --packing_samples
    # --nll_loss_coef (Regularization with NLL loss)


if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi