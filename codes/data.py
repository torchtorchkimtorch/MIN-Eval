from .utils import yaml_parser, make_chat_template, parse_answer
from transformers import AutoTokenizer
import os


class DataLoader:
    def __init__(self, args):
        self.tasks = args.tasks
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        self.model_name = args.model
        self.system_prompt = args.system_prompt
        self.is_reasoning = args.is_reasoning
        self.debug = args.debug
        self.boxed_prompt = args.boxed_prompt

    def load_data(self, task_full_name):
        parts = task_full_name.split("-", 1)
        task_name = parts[0]
        task_specific_name = parts[1] if len(parts) > 1 else None

        tasks_dir = "./tasks"
        if task_name not in os.listdir(tasks_dir):
            raise ValueError(f"Task {task_name} not found in the tasks directory.")

        if task_specific_name:
            yaml_path = os.path.join(
                tasks_dir, task_name, f"{task_name}-{task_specific_name}.yaml"
            )
        else:
            yaml_path = os.path.join(tasks_dir, task_name, f"{task_name}.yaml")

        if os.path.exists(yaml_path):
            task_config = yaml_parser(yaml_path)
            if task_config["repo_type"] == "hf":
                dataset = self.load_from_hf(
                    path=task_config["path"],
                    subset=task_config["subset"],
                    split=task_config["split"],
                )
                return dataset, task_config
            else:
                raise ValueError(
                    f"Unsupported repo type: {task_config.get('data',{}).get('repo_type','N/A')}"
                )
        else:
            raise ValueError(
                f"Unsupported repo type: {task_config.get('data',{}).get('repo_type','N/A')}"
            )

    @staticmethod
    def load_from_hf(path, subset, split):
        from datasets import load_dataset

        dataset = load_dataset(path, subset, split=split)
        return dataset

    def load_all_data(self):
        for task in self.tasks:
            data, task_config = self.load_data(task)
            question_column = task_config.get("question_column", "question")
            answer_column = task_config.get("answer_column", "answer")
            if len(question_column) == 1 and len(answer_column) == 1:
                if self.boxed_prompt:
                    data = data.map(
                        lambda x: {
                            "question": x[question_column[0]] + " Put your final answer within \\boxed{}.",
                            "answer": x[answer_column[0]],
                        }
                    )
                else:
                    data = data.map(
                        lambda x: {
                            "question": x[question_column[0]],
                            "answer": x[answer_column[0]],
                        }
                    )
                if self.is_reasoning:
                    chat_templates = [
                        self.tokenizer.apply_chat_template(
                            make_chat_template(self.system_prompt, sample["question"]),
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=True,
                        )
                        for sample in data
                    ]

                else:
                    chat_templates = [
                        self.tokenizer.apply_chat_template(
                            make_chat_template(self.system_prompt, sample["question"]),
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                        for sample in data
                    ]
                if not task_config["only_answer"]:
                    data = data.map(lambda batch: parse_answer(batch, task), batched=True)
                if self.debug and len(chat_templates) > 5 :
                    yield (chat_templates[:5], data["answer"][:5], task_config)
                else:
                    yield (chat_templates, data["answer"], task_config)
            else:
                # Should handle multiple answer & question columns 
                pass
