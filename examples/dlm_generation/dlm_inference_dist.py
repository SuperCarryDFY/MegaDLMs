import os
import ray
import torch 
import random
import logging
import argparse
import numpy as np 
from tqdm import tqdm 
from ray.util.actor_pool import ActorPool
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# set seed 
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [DLMGeneration] - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def init_model_tokenizer(model_dir_path):
    tokenizer = AutoTokenizer.from_pretrained(model_dir_path)
    model = AutoModelForCausalLM.from_pretrained(
                model_dir_path,
                torch_dtype="auto",
                trust_remote_code=True,
                attn_implementation="flash_attention_2"
            ).to('cuda')
    return model, tokenizer

def init_model_inputs(tokenizer, batch_size):
    cls_token = tokenizer.cls_token_id
    init_input_ids = torch.tensor([cls_token]).to('cuda').unsqueeze(0).repeat(batch_size, 1)
    init_attention_mask = torch.tensor([1]).to('cuda').unsqueeze(0).repeat(batch_size, 1)
    model_inputs = {
        "input_ids": init_input_ids,
        "attention_mask": init_attention_mask,}
    return model_inputs

@ray.remote(num_gpus=1)
class DLMGenerationActor:
    def __init__(self, model_dir_path, temperature=1.0, cfg=0.0, remasking="random", verbose=True):
        self.gpu_id = torch.cuda.current_device()
        self.process_id = os.getpid()
        
        self.model, self.tokenizer = init_model_tokenizer(model_dir_path)
        self.temperature = temperature
        self.cfg = cfg
        self.remasking = remasking
        self.verbose = verbose
        
        seed = 42 + self.gpu_id
        logger.warning(f"Actor initialized - GPU: {self.gpu_id}, PID: {self.process_id}, Seed: {seed}")
        set_seed(seed)    

    def generate(self, batch_size, seq_length):
        model_inputs = init_model_inputs(self.tokenizer, batch_size)
        generated_ids = self.model.generate(
            **model_inputs,
            temperature = self.temperature,
            cfg = self.cfg,
            remasking = self.remasking,
            inference_block_size = seq_length,
            sample_steps = seq_length,
            max_new_tokens = seq_length,
            verbose = self.verbose
        )
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        response = [i.replace(" ", "") for i in response]

        return response

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir_path", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cfg", type=float, default=0.0)
    parser.add_argument("--remasking", type=str, default="random", choices=["random", "low_confidence"])
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=5)
    args = parser.parse_args()

    # check the output path
    exp_name = args.model_dir_path.split('/')[-3]
    ckpt_name = args.model_dir_path.split('/')[-2]
    output_path = f"output/{exp_name}/generation_output/{ckpt_name}/sequences_t{args.temperature}_remasking{args.remasking}.fasta"
    if os.path.exists(output_path):
        logger.info(f"Output path {output_path} already exists. Skipping...")
        exit()

    num_gpus = torch.cuda.device_count()
    logger.info(f"Detected {num_gpus} GPUs. Spawning actors...")
    
    actors = [DLMGenerationActor.remote(args.model_dir_path, args.temperature, args.cfg, args.remasking, args.verbose) for _ in range(num_gpus)]
    pool = ActorPool(actors)
    chunk_num = args.num_samples // args.batch_size
    seq_length_list = [random.randint(100, 400) for _ in range(chunk_num)]
    pbar = tqdm(total=chunk_num, desc="Generating samples", unit="batch")
    all_results = []
    logger.info(f"Generating samples...")
    for results in pool.map(lambda actor, value: actor.generate.remote(args.batch_size, value), seq_length_list):
        all_results.extend(results)
        pbar.update(1)
    pbar.close()

    # save the results
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    with open(output_path, 'w') as f:
        for idx, result in enumerate(all_results):
            f.write(f">sequence_{idx}\n{result}\n")
    
    logger.info(f"Generation completed. Saved {len(all_results)} sequences to {output_path}")