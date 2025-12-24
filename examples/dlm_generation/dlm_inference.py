from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch 


# todo: point the model_dir_path to the dir containing your converted hf checkpoint.
model_dir_path = 'ckpts/cache/difflm/converted_checkpoints/99-1218_3-dlm_training_ur50_unconditional/ckptstep_70000/hf'
# prompts = ["", ""]

tokenizer = AutoTokenizer.from_pretrained(model_dir_path)
cls_token = tokenizer.cls_token_id
eos_token = tokenizer.eos_token_id
# config = AutoConfig.from_pretrained(model_dir_path)
# model = AutoModelForCausalLM.from_config(config).to('cuda')
model = AutoModelForCausalLM.from_pretrained(
            model_dir_path,
            torch_dtype="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        ).to('cuda')
init_input_ids = torch.tensor([cls_token]).to('cuda').unsqueeze(0)
model.config.eos_token_id = eos_token # should be 2
init_attention_mask = torch.tensor([1]).to('cuda').unsqueeze(0)
# model_inputs = tokenizer(prompts, return_tensors="pt", padding_side='left', padding=True).to('cuda')
# print(model_inputs)
model_inputs = {
    "input_ids": init_input_ids,
    "attention_mask": init_attention_mask,}

print(model_inputs)
generated_ids = model.generate(
    **model_inputs,
    temperature = 0.7,
    cfg = 0.0,
    # remasking = "random",
    # remasking="low_confidence",  # May cause repetition issues
    remasking="random",  # Use entropy-based remasking to avoid repetition
    inference_block_size = 200,
    sample_steps = 200,
    max_new_tokens = 200,
    verbose=True
    )

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
response = [i.replace(" ", "") for i in response]

print(response)