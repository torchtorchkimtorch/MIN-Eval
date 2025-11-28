import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
from .vllm import VLLM
from setproctitle import setproctitle
import json

from .utils import args_exp_parser, eval_config_printer, code_maker
from .data import DataLoader
from .verify import Verify

class Eval:
    def __init__(self, args):

        # args for inference
        self.model = args_exp_parser(args, "model")
        self.is_reasoning = args_exp_parser(args, "is_reasoning")
        self.tasks = args_exp_parser(args, "tasks")
        self.device = args_exp_parser(args, "device")
        self.max_batch_size = args_exp_parser(args, "max_batch_size")
        self.tensor_parallel = args_exp_parser(args, "tensor_parallel")
        self.temperature = args_exp_parser(args, "temperature")
        self.top_k = args_exp_parser(args, "top_k")
        self.top_p = args_exp_parser(args, "top_p")
        self.max_tokens = args_exp_parser(args, "max_tokens")
        self.seed = args_exp_parser(args, "seed")
        self.system_prompt = args_exp_parser(args, "system_prompt")
        self.boxed_prompt = args_exp_parser(args, "boxed_prompt")
        self.n_repetitions = args_exp_parser(args, "n_repetitions")

        # args for etc
        self.setproctitle = args_exp_parser(args, "setproctitle")
        self.debug = args_exp_parser(args, "debug")

        # args for output
        self.eval_type = args_exp_parser(args, "eval_type")
        self.output_dir = args_exp_parser(args, "output_dir")
        self.save_logs = args_exp_parser(args, "save_logs")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)

    def __call__(self):
        setproctitle(self.setproctitle)
        eval_config_printer(self)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        data_loader = DataLoader(self)
        self.datas, self.answers, self.configs = zip(*data_loader.load_all_data())
        
        vllm = VLLM(
            model=self.model,
            device=self.device,
            tensor_parallel=self.tensor_parallel,
            max_batch_size=self.max_batch_size,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            seed=self.seed,
            n_repetitions=self.n_repetitions,
        )
        
        for i, data in enumerate(self.datas):
            result_code = code_maker()
            outputs = vllm.generate(data)

            # For pass@k evaluation
            total_problems = 0
            correct_problems = 0  # pass@k (at least one correct among k attempts)
            all_correct = 0  # Total correct answers across all repetitions

            # Each output contains n_repetitions completions in output.outputs list
            for j, output in enumerate(outputs):
                total_problems += 1
                curr_answer = self.answers[i][j]

                # Check all n_repetitions for this problem
                problem_correct = False

                # output.outputs is a list of CompletionOutput objects (length = n_repetitions)
                for completion_output in output.outputs:
                    generated_text = completion_output.text
                    target = Verify(self.configs[i], self.boxed_prompt, curr_answer, generated_text)
                    result = target.tf_verify()

                    if result == 1:
                        problem_correct = True
                        all_correct += 1

                if problem_correct:
                    correct_problems += 1

            # Calculate metrics
            pass_at_k = correct_problems / total_problems if total_problems > 0 else 0
            avg_accuracy = all_correct / (total_problems * self.n_repetitions) if total_problems > 0 else 0

            pass_at_k_str = f"{pass_at_k:.4f}"
            avg_acc_str = f"{avg_accuracy:.4f}"

            print(f"Pass@{self.n_repetitions}: {pass_at_k_str}")
            print(f"Average Accuracy: {avg_acc_str}")

            # Create output directory if it doesn't exist
            output_path = os.path.join("/mnt/raid6/mhkim0929/fitr/MIN-Eval/results",
                                      f"{self.configs[i]['task']}",
                                      os.path.basename(self.model))
            os.makedirs(output_path, exist_ok=True)

            # Save results with timestamp and n_repetitions in filename
            output_file = os.path.join(output_path, f"pass_at_{self.n_repetitions}_{result_code}.json")

            with open(output_file, "w") as f:
                json.dump({
                    "model": self.model,
                    "task": self.configs[i]['task'],
                    "hyperparameters": {
                        "temperature": self.temperature,
                        "top_k": self.top_k,
                        "top_p": self.top_p,
                        "max_tokens": self.max_tokens,
                        "seed": self.seed,
                        "n_repetitions": self.n_repetitions,
                    },
                    "result": {
                        "pass_at_k": pass_at_k_str,
                        "average_accuracy": avg_acc_str,
                        "correct_problems": correct_problems,
                        "total_problems": total_problems,
                        "total_correct_answers": all_correct,
                        "total_answers": total_problems * self.n_repetitions,
                    }
                }, f)
