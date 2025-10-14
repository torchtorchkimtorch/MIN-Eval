from tqdm import tqdm
import os
from vllm import LLM, SamplingParams


class VLLM:
    def __init__(
        self,
        model,
        device,
        tensor_parallel,
        max_batch_size,
        temperature,
        top_k,
        top_p,
        max_tokens,
        seed,
    ):
        self.model = model
        self.device = device
        self.tensor_parallel = tensor_parallel
        self.max_batch_size = max_batch_size
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.seed = seed

    def generate(self, tokenized_templates):
        if not self.tensor_parallel:
            try:
                llm = LLM(model=self.model)
            except Exception as e:
                raise ValueError(
                    "Model not found. Please check the model name or path."
                )
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(self.device)
            try:
                llm = LLM(
                    model=self.model,
                    tensor_parallel_size=len(self.device),
                    max_num_seqs=self.max_batch_size,
                )
            except Exception as e:
                raise ValueError(
                    "Model not found. Please check the model name or path."
                )

        BATCH_SIZE = self.max_batch_size
        num_batches = (len(tokenized_templates) + BATCH_SIZE - 1) // BATCH_SIZE
        sampling_params = SamplingParams(
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            seed=self.seed,
        )

        all_outputs = []
        for i in tqdm(range(num_batches), desc="Inference Progress"):
            batch_prompts = tokenized_templates[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
            outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)
            all_outputs.extend(outputs)
        return all_outputs
