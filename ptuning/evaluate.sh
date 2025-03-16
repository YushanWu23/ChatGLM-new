PRE_SEQ_LEN=300
CHECKPOINT=adgen-chatglm2-6b-pt-300-1.8e-2-new
STEP=1200
NUM_GPUS=1

torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS main.py \
    --do_predict \
    --validation_file data/val_new.json \
    --test_file data/test_new.json \
    --overwrite_cache \
    --prompt_column content \
    --response_column summary \
    --model_name_or_path /autodl-fs/data/ChatGLM \
    --ptuning_checkpoint ./output/$CHECKPOINT/checkpoint-$STEP \
    --output_dir ./output/$CHECKPOINT/new \
    --overwrite_output_dir \
    --max_source_length 300 \
    --max_target_length 1024 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4 \
    --report_to tensorboard \