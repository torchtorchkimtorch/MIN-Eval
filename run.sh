DEVICES=4

CUDA_VISIBLE_DEVICES=${DEVICES} python run.py \
--model /mnt/raid6/mhkim0929/fitr/train/output/2e-5_fim \
--tasks gsm8k \
--device 4 \
--max_batch_size 1 \
--eval_type generation \
--is_reasoning True \
--setproctitle MIN-Eval \
--output_dir ./results \
--debug False \
--save_logs False \
--tensor_parallel False \
--temperature 0.6 \
--top_p 0.95 ;