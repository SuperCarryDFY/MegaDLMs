from transformers import AutoTokenizer, AutoModelForCausalLM
import torch 
# todo: point the model_dir_path to the dir containing your converted hf checkpoint.
model_dir_path = 'ckpts/cache/difflm/converted_checkpoints/dlm_training/ckptstep_90000/hf'
# prompts = ["", ""]

tokenizer = AutoTokenizer.from_pretrained(model_dir_path)
cls_token = tokenizer.cls_token_id

model = AutoModelForCausalLM.from_pretrained(
            model_dir_path,
            torch_dtype="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        ).to('cuda')
init_input_ids = torch.tensor([cls_token]).to('cuda').unsqueeze(0).repeat(5, 1)
init_attention_mask = torch.tensor([1]).to('cuda').unsqueeze(0).repeat(5, 1)
# model_inputs = tokenizer(prompts, return_tensors="pt", padding_side='left', padding=True).to('cuda')
# print(model_inputs)
model_inputs = {
    "input_ids": init_input_ids,
    "attention_mask": init_attention_mask,}

print(model_inputs)
generated_ids = model.generate(
    **model_inputs,
    temperature = 2.0,
    cfg = 0.0,
    remasking = "low_confidence",
    inference_block_size = 256,
    sample_steps = 256,
    max_new_tokens = 256,
    verbose=True
    )

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

print(response)