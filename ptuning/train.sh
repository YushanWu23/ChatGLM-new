PRE_SEQ_LEN=300
LR=1.8e-2
NUM_GPUS=1

torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS main.py \
    --do_train \
    --train_file data/train_new.json \
    --validation_file data/val_new.json \
    --preprocessing_num_workers 12 \
    --prompt_column content \
    --response_column summary \
    --overwrite_cache \
    --model_name_or_path /autodl-fs/data/ChatGLM \
    --output_dir output/adgen-chatglm2-6b-pt-$PRE_SEQ_LEN-$LR-new \
    --report_to tensorboard \
    --overwrite_output_dir \
    --max_source_length 300 \
    --max_target_length 1024 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --predict_with_generate \
    --max_steps 4000 \
    --logging_steps 50 \
    --save_steps 100 \
    --save_total_limit 2\
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4 \
    --weight_decay 0.012 \
    --dropout_rate 0.12 \
    --lr_scheduler_type cosine_with_restarts \
    --num_cycles 2 \
    --warmup_ratio 0.1 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --early_stopping_patience 6 \
    --load_best_model_at_end \
    --metric_for_best_model rouge-1 \
    --save_strategy steps \
    --max_grad_norm 1.0 \
    --fp16 \