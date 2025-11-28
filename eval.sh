# MODEL_PATH=/mnt/raid6/mhkim0929/fitr/train/output/2e-5_20K_fim_0_5_SPM_variant
DEVICE=0

CUDA_VISIBLE_DEVICES=$DEVICE python run.py \
--model /mnt/raid6/mhkim0929/fitr/train/output/WBL_2e-5_20K_fim_0_5_PSM \
--tasks gsm8k \
--max_batch_size 32 \
--eval_type generation \
--is_reasoning True \
--setproctitle MIN-Eval \
--output_dir ./results \
--debug False \
--save_logs False \
--tensor_parallel False \
--temperature 0.6 \
--max_tokens 4096 \
--device $DEVICE \
--top_p 0.95 \
--n_repetitions 1 ;

CUDA_VISIBLE_DEVICES=$DEVICE python run.py \
--model /mnt/raid6/mhkim0929/fitr/train/output/WBL_2e-5_20K_onlychat \
--tasks gsm8k \
--max_batch_size 32 \
--eval_type generation \
--is_reasoning True \
--setproctitle MIN-Eval \
--output_dir ./results \
--debug False \
--save_logs False \
--tensor_parallel False \
--temperature 0.6 \
--max_tokens 4096 \
--device $DEVICE \
--top_p 0.95 \
--n_repetitions 1 ;

CUDA_VISIBLE_DEVICES=$DEVICE python run.py \
--model /mnt/raid6/mhkim0929/fitr/train/output/WBL_2e-5_10K_fim_0_5_PSM \
--tasks gsm8k \
--max_batch_size 32 \
--eval_type generation \
--is_reasoning True \
--setproctitle MIN-Eval \
--output_dir ./results \
--debug False \
--save_logs False \
--tensor_parallel False \
--temperature 0.6 \
--max_tokens 4096 \
--device $DEVICE \
--top_p 0.95 \
--n_repetitions 1 ;

CUDA_VISIBLE_DEVICES=$DEVICE python run.py \
--model /mnt/raid6/mhkim0929/fitr/train/output/WBL_2e-5_10K_onlychat \
--tasks gsm8k \
--max_batch_size 32 \
--eval_type generation \
--is_reasoning True \
--setproctitle MIN-Eval \
--output_dir ./results \
--debug False \
--save_logs False \
--tensor_parallel False \
--temperature 0.6 \
--max_tokens 4096 \
--device $DEVICE \
--top_p 0.95 \
--n_repetitions 1 ;

CUDA_VISIBLE_DEVICES=$DEVICE python run.py \
--model /mnt/raid6/mhkim0929/fitr/train/output/WBL_2e-5_20K_fim_0_5_PSM \
--tasks aime2025 \
--max_batch_size 8 \
--eval_type generation \
--is_reasoning True \
--setproctitle MIN-Eval \
--output_dir ./results \
--debug False \
--save_logs False \
--tensor_parallel False \
--temperature 0.6 \
--max_tokens 16384 \
--device $DEVICE \
--top_p 0.95 \
--n_repetitions 1 ;

CUDA_VISIBLE_DEVICES=$DEVICE python run.py \
--model /mnt/raid6/mhkim0929/fitr/train/output/WBL_2e-5_20K_onlychat \
--tasks aime2025 \
--max_batch_size 8 \
--eval_type generation \
--is_reasoning True \
--setproctitle MIN-Eval \
--output_dir ./results \
--debug False \
--save_logs False \
--tensor_parallel False \
--temperature 0.6 \
--max_tokens 16384 \
--device $DEVICE \
--top_p 0.95 \
--n_repetitions 1 ;

CUDA_VISIBLE_DEVICES=$DEVICE python run.py \
--model /mnt/raid6/mhkim0929/fitr/train/output/WBL_2e-5_10K_fim_0_5_PSM \
--tasks aime2025 \
--max_batch_size 8 \
--eval_type generation \
--is_reasoning True \
--setproctitle MIN-Eval \
--output_dir ./results \
--debug False \
--save_logs False \
--tensor_parallel False \
--temperature 0.6 \
--max_tokens 16384 \
--device $DEVICE \
--top_p 0.95 \
--n_repetitions 1 ;

CUDA_VISIBLE_DEVICES=$DEVICE python run.py \
--model /mnt/raid6/mhkim0929/fitr/train/output/WBL_2e-5_10K_onlychat \
--tasks aime2025 \
--max_batch_size 8 \
--eval_type generation \
--is_reasoning True \
--setproctitle MIN-Eval \
--output_dir ./results \
--debug False \
--save_logs False \
--tensor_parallel False \
--temperature 0.6 \
--max_tokens 16384 \
--device $DEVICE \
--top_p 0.95 \
--n_repetitions 1 ;