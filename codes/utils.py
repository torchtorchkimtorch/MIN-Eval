from typing import List, Union


# function which parses command line arguments for evaluation
def args_exp_parser(args, arg_name) -> Union[str, List[str], int]:
    if arg_name == "model":
        result: str = args.model
        return result

    elif arg_name == "tasks":
        if "," in args.tasks:
            parsed_tasks: List[str] = []
            tasks = args.tasks.split(",")
            for task in tasks:
                if task != " ":
                    parsed_tasks.append(task.strip())
            return parsed_tasks
        else:
            result: List[str] = [args.tasks.strip()]
            return result

    elif arg_name == "device":
        if "," in args.device:
            parsed_devices: List[str] = []
            devices = args.device.split(",")
            for device in devices:
                if device != " ":
                    if device.isdecimal():
                        parsed_devices.append(device.strip())
                    else:
                        raise ValueError(f"Invalid device argument: {device}")
            return parsed_devices
        else:
            result: List[str] = [args.device.strip()]
            return result

    elif arg_name == "max_batch_size":
        if str(args.max_batch_size).isdecimal():
            result: int = int(args.max_batch_size)
            return result
        else:
            raise ValueError(f"Invalid batch size argument: {args.max_batch_size}")

    elif arg_name == "eval_type":
        # Should implement the case of generation with multi answers
        if args.eval_type in ["generation", "logit"]:
            result: str = args.eval_type
            return result
        else:
            raise ValueError(f"Invalid eval type argument: {args.eval_type}")

    elif arg_name == "is_reasoning":
        if isinstance(args.is_reasoning, bool):
            return args.is_reasoning
        else:
            raise ValueError(f"Invalid is_reasoning argument: {args.is_reasoning}")

    elif arg_name == "setproctitle":
        if isinstance(args.setproctitle, str):
            return args.setproctitle
        else:
            raise ValueError(f"Invalid setproctitle argument: {args.setproctitle}")

    elif arg_name == "output_dir":
        if isinstance(args.output_dir, str):
            return args.output_dir
        else:
            raise ValueError(f"Invalid output_dir argument: {args.output_dir}")

    elif arg_name == "debug":
        if isinstance(args.debug, bool):
            return args.debug
        else:
            raise ValueError(f"Invalid debug argument: {args.debug}")

    elif arg_name == "save_logs":
        if isinstance(args.save_logs, bool):
            return args.save_logs
        else:
            raise ValueError(f"Invalid save_logs argument: {args.save_logs}")

    elif arg_name == "tensor_parallel":
        if isinstance(args.tensor_parallel, bool):
            if args.tensor_parallel and "," not in args.device:
                raise ValueError("Tensor parallelism requires multiple devices.")
            return args.tensor_parallel
        else:
            raise ValueError(
                f"Invalid tensor_parallel argument: {args.tensor_parallel}"
            )

    elif arg_name == "temperature":
        if isinstance(args.temperature, float):
            return args.temperature
        else:
            raise ValueError(f"Invalid temperature argument: {args.temperature}")

    elif arg_name == "top_p":
        if isinstance(args.top_p, float):
            return args.top_p
        else:
            raise ValueError(f"Invalid top_p argument: {args.top_p}")

    elif arg_name == "top_k":
        if isinstance(args.top_k, int):
            return args.top_k
        else:
            raise ValueError(f"Invalid top_k argument: {args.top_k}")

    elif arg_name == "max_tokens":
        if isinstance(args.max_tokens, int):
            return args.max_tokens
        else:
            raise ValueError(f"Invalid max_tokens argument: {args.max_tokens}")

    elif arg_name == "seed":
        if isinstance(args.seed, int):
            return args.seed
        else:
            raise ValueError(f"Invalid seed argument: {args.seed}")

    elif arg_name == "system_prompt":
        if isinstance(args.system_prompt, str):
            return args.system_prompt
        else:
            raise ValueError(f"Invalid system_prompt argument: {args.system_prompt}")
    
    elif arg_name == "boxed_prompt":
        if isinstance(args.boxed_prompt, bool):
            return args.boxed_prompt
        else:
            raise ValueError(f"Invalid boxed_prompt argument: {args.boxed_prompt}")

    elif arg_name == "n_repetitions":
        if isinstance(args.n_repetitions, int) and args.n_repetitions > 0:
            return args.n_repetitions
        else:
            raise ValueError(f"Invalid n_repetitions argument: {args.n_repetitions}")

    else:
        raise ValueError(f"Not Implemented: {arg_name}")


# function which convert string to boolean, especially for argparse library paradox
def str2bool(v):
    if isinstance(v, bool):
        return v

    if v.lower() in ("true", "True"):
        return True

    elif v.lower() in ("false", "False"):
        return False


# function which prints the evaluation configuration while running the evaluation
def eval_config_printer(args):
    print("=============================================")
    print("             Eval Configuration")
    print("---------------------------------------------")
    if args.is_reasoning:
        print(f"       Reasoning Mode for hybrid model")
        print("---------------------------------------------")
    if args.debug:
        print("                 Debug Mode")
    if args.debug:
        print("---------------------------------------------")
    print(f"Process Title: {args.setproctitle}")
    print(f"Model: {args.model}")
    print(f"Tasks: {', '.join(args.tasks)}")
    print(f"Device: {', '.join(args.device)}")
    print(f"Batch Size: {args.max_batch_size}")
    print(f"Eval Type: {args.eval_type}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Save Logs: {args.save_logs}")
    print("=============================================")


def yaml_parser(yaml_path):
    import yaml

    with open(yaml_path, "r") as file:
        config = yaml.safe_load(file)
    task = config["info"]["tasks"]
    type = config["info"]["type"]
    default_eval_type = config["info"]["default_eval_type"]
    repo_type = config["data"]["repo_type"]
    path = config["data"]["path"]
    subset = config["data"]["subset"]
    split = config["data"]["split"]
    question_column = config["data"]["question_column"]
    answer_column = config["data"]["answer_column"]
    only_answer = config["data"]["only_answer"]
    return {
        "task": task,
        "type": type,
        "default_eval_type": default_eval_type,
        "repo_type": repo_type,
        "path": path,
        "subset": subset,
        "split": split,
        "question_column": question_column,
        "answer_column": answer_column,
        "only_answer": only_answer,
    }


def make_chat_template(system_prompt, question):
    if "gemma" not in system_prompt.lower():
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
    else:
        return [{"role": "user", "content": question}]


def answerParser(task, answer):
    if task in ["gsm8k", "aime2025"]:
        return answer.split("####")[-1].strip()
    else:
        raise NotImplementedError(f"function answerParser in utils.py was not implemented for task: {task}")


def parse_answer(batch, task):
    return {"answer": [answerParser(task, ans) for ans in batch["answer"]]}

def code_maker():
    from datetime import datetime

    result_code = str(datetime.now())
    result_code = result_code.replace("-", "")
    result_code = result_code.replace(" ", "")
    result_code = result_code.replace(":", "")
    result_code = result_code.split(".")[0]
    return result_code
