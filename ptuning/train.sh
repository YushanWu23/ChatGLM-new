PRE_SEQ_LEN=512
LR=1e-3
NUM_GPUS=1

torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS main.py \
    --do_train \
    --train_file data/train_new.json \
    --validation_file data/val_new.json \
    --preprocessing_num_workers 4 \
    --prompt_column content \
    --response_column summary \
    --overwrite_cache \
    --model_name_or_path /autodl-fs/data/ChatGLM \
    --output_dir output/adgen-chatglm2-6b-pt-$PRE_SEQ_LEN-$LR \
    --report_to tensorboard \
    --overwrite_output_dir \
    --max_source_length 300 \
    --max_target_length 1024 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --predict_with_generate \
    --max_steps 5000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4 \
    --num_train_epochs 100 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end \
    --weight_decay 0.01 \
    --dropout_rate 0.1 \
    --early_stopping_patience 5 \
