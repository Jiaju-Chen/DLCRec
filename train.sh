export NCCL_IB_DISABLE=1
## Note: change '--train_data_path --val_data_path --output_dir' for different tasks
## For example, use different training sets for each task:
## For task GP, train_1000_GP.json
## For task GF, train_1000_BERT_GF.json, train_1000_BERT_GFd.json, train_1000_BERT_GFn.json (the latter two for data augmentation)
## For task IP, train_1000_BERT_IP.json, train_1000_BERT_IPd.json, train_1000_BERT_IPn.json (the latter two for data augmentation)

dataset='movie_sgcate'
for seed in 1
do
    for lr in 1e-3
    do
        for dropout in 0.05    
        do
            for sample in -1
            do
                echo "lr: $lr, dropout: $dropout, seed: $seed,"
                python train.py \
                    --base_model "../Meta-Llama-3-8B-Instruct/" \
                    --train_data_path "['./data/$dataset/train_1000_GP.json']"   \
                    --val_data_path "['./data/$dataset/valid_1000_GP.json']" \
                    --output_dir ./model/${dataset}/BERT_GP_vlGP_ep25_${lr} \
                    --batch_size 250  \
                    --micro_batch_size 4 \
                    --num_epochs 25 \
                    --learning_rate $lr \
                    --cutoff_len 768 \
                    --lora_r 8 \
                    --lora_alpha 16\
                    --lora_dropout $dropout \
                    --lora_target_modules '[q_proj,v_proj]' \
                    --train_on_inputs \
                    --group_by_length \
                    --resume_from_checkpoint None \
                    --seed $seed \
                    --sample $sample
            done    
        done
    done
done
