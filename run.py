import argparse
from codes.main import Eval
from codes.utils import str2bool

parser = argparse.ArgumentParser()

# required
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--tasks", type=str, required=True)
parser.add_argument("--device", type=str, required=True)

# non-required
parser.add_argument("--max_batch_size", type=int, default=1)
parser.add_argument("--eval_type", type=str, default="generation", choices=["generation", "logit"])
parser.add_argument("--is_reasoning", type=str2bool, default=False)
parser.add_argument("--setproctitle", type=str, default="MIN-Eval")
parser.add_argument("--output_dir", type=str, default="./results/")
parser.add_argument("--debug", type=str2bool, default=False)
parser.add_argument("--save_logs", type=str2bool, default=False)
parser.add_argument("--tensor_parallel", type=str2bool, default=False)
parser.add_argument("--temperature", type=float, default=0.6)
parser.add_argument("--top_p", type=float, default=0.95)
parser.add_argument("--top_k", type=int, default=20)
parser.add_argument("--max_tokens", type=int, default=4096)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--system_prompt", type=str, default="You are a helpful assistant.")
parser.add_argument("--boxed_prompt", type=str2bool, default=True)
parser.add_argument("--n_repetitions", type=int, default=1, help="Number of responses to generate per prompt (for pass@k evaluation)")

args = parser.parse_args()
eval = Eval(args)
result = eval()
