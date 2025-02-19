torchrun --nproc_per_node 2 getemb.py \
    --ckpt_dir llama-2-13b-chat/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 1024 --max_batch_size 1 \
    --dataset webqsp