import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
from .vllm import VLLM
from setproctitle import setproctitle

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
        )
        
        for i, data in enumerate(self.datas):
            result_code = code_maker()
            outputs = vllm.generate(data)
            total = 0
            correct = 0
            for j, output in enumerate(outputs):
                total += 1
                prompt = output.prompt
                generated_text = output.outputs[0].text
                curr_answer = self.answers[i][j]
                target = Verify(self.configs[i], self.boxed_prompt, curr_answer, generated_text)
                result = target.tf_verify()
                correct += result
            print(correct / total)
            with open(os.path.join(self.output_dir, f"/{self.configs[i]['task']}/",f"{self.model}",f"_{result_code}.json"), "w") as f:
                # result code
                
                
