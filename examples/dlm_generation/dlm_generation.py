import torch 
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
import numpy as np
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TopPLogitsWarper,
)
import argparse
import math

# from accelerate.utils import set_seed
logger = logging.getLogger(__name__)


# set_seed(42)

@torch.no_grad()
def dlm_generation(
    dlm, 
    tokenizer,
    # generation_config: GenerationConfig = None,
    **kwargs,
    ):

    
    def get_num_transfer_tokens(mask_index, steps):
        '''
        In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
        Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
        the expected number of tokens transitioned at each step should be consistent.

        This function is designed to precompute the number of tokens that need to be transitioned at each step.
        '''
        mask_num = mask_index.sum(dim=1, keepdim=True)

        base = mask_num // steps
        remainder = mask_num % steps

        num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

        for i in range(mask_num.size(0)):
            num_transfer_tokens[i, :remainder[i]] += 1

        return num_transfer_tokens
    '''
    model_kwargs:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK].
    '''
    generation_config = {
    "eos_token_id": 0,
    "max_new_tokens": 300
    }
    kwargs = {'input_ids': torch.tensor([[0]], device='cuda'), 
            'attention_mask': torch.tensor([[1]], device='cuda'), 
            'temperature': 1.0, 'cfg': 0.0, 'remasking': 'entropy', 'inference_block_size': 300, 'sample_steps': 300, 'max_new_tokens': 300, 'verbose': True}
    model_kwargs = {'input_ids': torch.tensor([[0]], device='cuda'), 
                'attention_mask': torch.tensor([[1]], device='cuda'), 
                'cfg': 0.0, 'remasking': 'low_confidence', 'inference_block_size': 300, 'sample_steps': 300, 'verbose': True}
    # generation_config, model_kwargs = super()._prepare_generation_config(generation_config, **kwargs)

    verbose = 'verbose' in model_kwargs and model_kwargs['verbose']
    prompt = model_kwargs['input_ids']
    steps=model_kwargs['sample_steps']
    block_length=model_kwargs['inference_block_size']
    cfg_scale=model_kwargs['cfg']
    remasking=model_kwargs['remasking']
    
    
    temperature=1
    top_p=1
    mask_id=tokenizer.mask_token_id

    if 'max_new_tokens' in kwargs:
        if 'max_length' in kwargs:
            logger.warning(
                f"Both `max_new_tokens` (={kwargs['max_new_tokens']}) and `max_length`(="
                f"{kwargs['max_length']}) seem to have been set. `max_new_tokens` will take precedence."
            )
        gen_length = kwargs['max_new_tokens']
    else:
        if 'max_length' in kwargs:
            gen_length = kwargs['max_length'] - prompt.shape[1]
        else:
            gen_length = dlm.config.max_position_embeddings - prompt.shape[1]
            logger.warning(
                f"None of the `max_new_tokens` and `max_length` seem to have been set. "
                f"Using `max_position_embeddings` ({dlm.config.max_position_embeddings})"
                f"- `input_ids.shape[1]` ({prompt.shape[1]}) = {gen_length} as maximum number of tokens to generate."
            )
            
    attention_mask = model_kwargs['attention_mask']
    attention_mask = F.pad(attention_mask, (0, gen_length), value=1)
    
    if cfg_scale > 0.:
        attention_mask = torch.cat([attention_mask, attention_mask], dim=0)
    
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, value=1)
    
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to("cuda")
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0, f"gen_length {gen_length} % block_length {block_length} != 0"
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0, f"steps {steps} % num_blocks {num_blocks} != 0"
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = dlm._forward(x_)
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = dlm._forward(x)

            if temperature == 0:
                x0 = torch.argmax(logits, dim=-1)
            else:
                batch_size, seq_len, vocab_size = logits.shape
                logits_flat = logits.view(-1, vocab_size)

                if top_p is not None and top_p < 1.0:
                    warper = TopPLogitsWarper(top_p)
                    logits_flat = warper(None, logits_flat)
                
                scaled_logits = logits_flat / temperature
                probs = F.softmax(scaled_logits, dim=-1)
                sampled_tokens = torch.multinomial(probs, num_samples=1)
                x0 = sampled_tokens.view(batch_size, seq_len)

            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

            seq = tokenizer.decode(x[0])
            print(seq)

    return x



@torch.no_grad()
def dlm_generation_v2(
    dlm, 
    tokenizer,
    # generation_config: GenerationConfig = None,
    **kwargs,
    ):

    
    def get_num_transfer_tokens(mask_index, steps):
        '''
        In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
        Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
        the expected number of tokens transitioned at each step should be consistent.

        This function is designed to precompute the number of tokens that need to be transitioned at each step.
        '''
        mask_num = mask_index.sum(dim=1, keepdim=True)

        base = mask_num // steps
        remainder = mask_num % steps

        num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

        for i in range(mask_num.size(0)):
            num_transfer_tokens[i, :remainder[i]] += 1

        return num_transfer_tokens
    '''
    model_kwargs:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK].
    '''
    generation_config = {
    "eos_token_id": 0,
    "max_new_tokens": 300
    }
    kwargs = {'input_ids': torch.tensor([[0]], device='cuda'), 
            'attention_mask': torch.tensor([[1]], device='cuda'), 
            'temperature': 1.0, 'cfg': 0.0, 'remasking': 'entropy', 'inference_block_size': 300, 'sample_steps': 300, 'max_new_tokens': 300, 'verbose': True}
    model_kwargs = {'input_ids': torch.tensor([[0]], device='cuda'), 
                'attention_mask': torch.tensor([[1]], device='cuda'), 
                'cfg': 0.0, 'remasking': 'low_confidence', 'inference_block_size': 300, 'sample_steps': 300, 'verbose': True}
    # generation_config, model_kwargs = super()._prepare_generation_config(generation_config, **kwargs)

    verbose = 'verbose' in model_kwargs and model_kwargs['verbose']
    prompt = model_kwargs['input_ids']
    steps=model_kwargs['sample_steps']
    block_length=model_kwargs['inference_block_size']
    cfg_scale=model_kwargs['cfg']
    remasking=model_kwargs['remasking']
    
    
    temperature=1
    top_p=1
    mask_id=tokenizer.mask_token_id

    if 'max_new_tokens' in kwargs:
        if 'max_length' in kwargs:
            logger.warning(
                f"Both `max_new_tokens` (={kwargs['max_new_tokens']}) and `max_length`(="
                f"{kwargs['max_length']}) seem to have been set. `max_new_tokens` will take precedence."
            )
        gen_length = kwargs['max_new_tokens']
    else:
        if 'max_length' in kwargs:
            gen_length = kwargs['max_length'] - prompt.shape[1]
        else:
            gen_length = dlm.config.max_position_embeddings - prompt.shape[1]
            logger.warning(
                f"None of the `max_new_tokens` and `max_length` seem to have been set. "
                f"Using `max_position_embeddings` ({dlm.config.max_position_embeddings})"
                f"- `input_ids.shape[1]` ({prompt.shape[1]}) = {gen_length} as maximum number of tokens to generate."
            )
            
    attention_mask = model_kwargs['attention_mask']
    attention_mask = F.pad(attention_mask, (0, gen_length), value=1)
    
    if cfg_scale > 0.:
        attention_mask = torch.cat([attention_mask, attention_mask], dim=0)
    
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, value=1)

    # 初始状态：全掩码
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to("cuda")
    x[:, :prompt.shape[1]] = prompt.clone()

    # 定义哪些位置是可以被 Mask 的（即生成的回答部分）
    # 如果是全局生成，maskable_mask 就是回答区域
    maskable_mask = torch.zeros_like(x, dtype=torch.bool)
    maskable_mask[:, prompt.shape[1]:] = True 
    
    # 获取需要生成的总长度
    total_gen_len = gen_length 

    for i in range(1, steps + 1):
        # 1. 前向预测 x0
        logits = dlm._forward(x)
        
        # 屏蔽特殊 Token，防止它们干扰置信度排序
        logits[..., [tokenizer.pad_token_id, tokenizer.mask_token_id]] = -float('inf')

        # 2. 采样当前的 x0 候选
        if temperature == 0:
            x0 = torch.argmax(logits, dim=-1)
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            batch_size, seq_len, vocab_size = probs.shape
            x0 = torch.multinomial(probs.view(-1, vocab_size), num_samples=1).view(batch_size, seq_len)

        # 3. 计算置信度 (Confidence)
        # 这里使用全局 softmax 概率作为分数
        all_probs = F.softmax(logits.to(torch.float64), dim=-1)
        confidence = torch.gather(all_probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)

        # 4. 关键：计算当前步应该保留多少个 Token
        # 线性调度：第 i 步保留 i/steps 的 token
        # 如果 i=100, steps=500, 则保留 20%，mask 掉 80%
        keep_rate = i / steps
        num_keep = int(total_gen_len * keep_rate)
        
        # 5. 全局排序与重掩码
        # 我们只给“可生成的区域”打分，Prompt 区域设为极高值不参与被 mask
        score_for_selection = torch.where(maskable_mask, confidence, torch.tensor(float('inf')).to(x.device))
        
        new_x = x.clone()
        for b in range(x.shape[0]):
            # 找到得分最低的 (total_gen_len - num_keep) 个位置
            num_to_mask = total_gen_len - num_keep
            if num_to_mask > 0:
                # 获取得分最低的索引
                _, val_idx = torch.topk(score_for_selection[b], k=num_to_mask, largest=False)
                
                # 先把 x0 的预测填进去
                temp_x0 = x0[b]
                # 然后把得分低的位置抹成 MASK
                temp_x0[val_idx] = mask_id
                new_x[b, prompt.shape[1]:] = temp_x0[prompt.shape[1]:]
            else:
                # 最后一步，全部保留
                new_x[b, prompt.shape[1]:] = x0[b, prompt.shape[1]:]

        x = new_x
        
        if verbose:
            print(f"Step {i}/{steps}: {tokenizer.decode(x[0, prompt.shape[1]:])}")

    return x


def sample_from_categorical(logits=None, temperature=1.0):
    if temperature:
        dist = torch.distributions.Categorical(logits=logits.div(temperature))
        tokens = dist.sample()
        scores = dist.log_prob(tokens)
    else:
        scores, tokens = logits.log_softmax(dim=-1).max(dim=-1)
    return tokens, scores


def stochastic_sample_from_categorical(
    logits=None, temperature=1.0, noise_scale=1.0
):
    gumbel_noise = -torch.log(
        -torch.log(torch.rand_like(logits) + 1e-8) + 1e-8
    )
    logits = logits + noise_scale * gumbel_noise
    tokens, scores = sample_from_categorical(logits, temperature)
    # scores, tokens = logits.log_softmax(dim=-1).max(dim=-1)
    return tokens, scores


@torch.no_grad()
def dlm_generation_v3(
    dlm, 
    tokenizer,
    seq_len=200, 
    sample_steps=200,
    temperature=1.0,
    cfg=0.0,
    verbose=True,
):
    init_seq = torch.full((1, seq_len + 2), tokenizer.mask_token_id, dtype=torch.long).to('cuda')
    init_seq[:, 0] = tokenizer.cls_token_id
    init_seq[:, -1] = tokenizer.eos_token_id

    steps=sample_steps
    cfg_scale=cfg

    # 确定哪些位置是可以操作的（生成的回答部分）
    maskable_mask = torch.zeros_like(init_seq, dtype=torch.bool)
    maskable_mask[:, 1:-1] = True 
    total_gen_len = seq_len 

    # --- 2. 迭代去噪过程 ---
    for t in range(1, steps + 1):
        # A. 前向计算 Logits
        if cfg_scale > 0.:
            un_x = init_seq.clone()
            init_seq_combined = torch.cat([init_seq, un_x], dim=0)
            logits_combined = dlm.forward(init_seq_combined).logits
            logits, un_logits = torch.chunk(logits_combined, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        else:
            logits = dlm.forward(init_seq).logits

        # B. 屏蔽干扰项（防止采样出特殊符号）
        # 蛋白质模型中通常屏蔽 Mask, Pad, Bos, Eos 等
        logits[..., [tokenizer.pad_token_id, tokenizer.mask_token_id]] = -float('inf')

        # C. 采样候选词 (Argmax 或 Multinomial)
        if temperature == 0:
            cur_tokens = torch.argmax(logits, dim=-1)
            # D. 计算置信度 (Confidence Score)
            # 使用 Log-Softmax 在数值上比原始概率更稳定，排序更精准
            log_probs = F.log_softmax(logits.to(torch.float64), dim=-1)
            cur_scores = torch.gather(log_probs, dim=-1, index=cur_tokens.unsqueeze(-1)).squeeze(-1)

        elif temperature > 0:
            # 采样
            probs = F.softmax(logits / temperature, dim=-1)
            b, l, v = probs.shape
            cur_tokens = torch.multinomial(probs.view(-1, v), num_samples=1).view(b, l)
            # D. 计算置信度 (Confidence Score)
            # 使用 Log-Softmax 在数值上比原始概率更稳定，排序更精准
            log_probs = F.log_softmax(logits.to(torch.float64), dim=-1)
            cur_scores = torch.gather(log_probs, dim=-1, index=cur_tokens.unsqueeze(-1)).squeeze(-1)

        else:
            noise_scale = 1.0
            cur_tokens, cur_scores = stochastic_sample_from_categorical(
                logits, temperature=0.0, noise_scale=noise_scale
            )

        # E. 全局排序与重掩码逻辑 (DPLM Reparam 核心)
        # 计算当前步应该保留的 Token 比例 (线性调度，也可以换成余弦)
        rate = 1 - (t / steps)
        # cutoff_len 是在生成区域内，当前步必须维持 MASK 状态的数量
        cutoff_len = int(total_gen_len * rate)

        # 准备排序分数：非生成区域设为极高分，确保不被选中
        _scores_for_topk = cur_scores.clone()
        _scores_for_topk[~maskable_mask] = 1000.0 

        new_x = init_seq.clone()
        for b in range(init_seq.shape[0]):
            # 找到全序列中 Score 最低的 cutoff_len 个位置
            # 这些位置不管之前是不是 Token，现在都要变成/保持 MASK
            _, lowest_k_idx = torch.topk(_scores_for_topk[b], k=cutoff_len, largest=False)
            
            # 构造这一步的布尔掩码
            is_lowest_score = torch.zeros(init_seq.shape[1], dtype=torch.bool, device=init_seq.device)
            is_lowest_score[lowest_k_idx] = True

            # --- 状态转移逻辑 ---
            # 1. 对于得分足够高的位置 (not in lowest_k)：
            #    如果它是 MASK，现在解锁变成新采样的 cur_tokens
            #    如果它已经是 Token，现在更新为 cur_tokens (允许修正)
            to_update = maskable_mask[b] & ~is_lowest_score
            new_x[b, to_update] = cur_tokens[b, to_update]

            # 2. 对于得分低的位置 (in lowest_k)：
            #    强制回退/保持为 MASK
            to_mask = maskable_mask[b] & is_lowest_score
            new_x[b, to_mask] = tokenizer.mask_token_id

        init_seq = new_x

        print(f"Step {t}/{steps}: {tokenizer.decode(init_seq[0])}")

    return init_seq



@torch.no_grad()
def dlm_generation_v4(
    dlm, 
    tokenizer,
    seq_len=200, 
    sample_steps=200,
    temperature=1.0,
    verbose=True,
):

    output_tokens = torch.full((1, seq_len + 2), tokenizer.mask_token_id).to('cuda')
    output_tokens[:, 0] = tokenizer.cls_token_id
    output_tokens[:, -1] = tokenizer.eos_token_id

    output_scores = torch.full_like(output_tokens, -1e9, dtype=torch.bfloat16)
    maskable_mask = torch.zeros_like(output_tokens, dtype=torch.bool)
    maskable_mask[:, 1:-1] = True 
    total_gen_len = seq_len 

    for t in range(1, sample_steps + 1):
        logits = dlm.forward(output_tokens).logits

        logits[..., tokenizer.mask_token_id] = -math.inf
        logits[..., tokenizer._token_to_id["X"]] = -math.inf
        logits[..., tokenizer.pad_token_id] = -math.inf
        logits[..., tokenizer.cls_token_id] = -math.inf
        logits[..., tokenizer.eos_token_id] = -math.inf

        if temperature == 0:
            cur_tokens = torch.argmax(logits, dim=-1)
            cur_scores = torch.gather(logits.log_softmax(dim=-1), dim=-1, index=cur_tokens.unsqueeze(-1)).squeeze(-1)
        elif temperature > 0:
            probs = F.softmax(logits / temperature, dim=-1)
            b, l, v = probs.shape
            cur_tokens = torch.multinomial(probs.view(-1, v), num_samples=1).view(b, l)
            cur_scores = torch.gather(probs.log(), dim=-1, index=cur_tokens.unsqueeze(-1)).squeeze(-1)
        else:
            noise_scale = 1.0
            cur_tokens, cur_scores = stochastic_sample_from_categorical(
                logits, temperature=0.0, noise_scale=noise_scale
            )
        # print(maskable_mask.dtype)
        # print(cur_scores.dtype)
        # print(output_scores.dtype)

        output_tokens.masked_scatter_(maskable_mask, cur_tokens[maskable_mask])
        output_scores.masked_scatter_(maskable_mask, cur_scores[maskable_mask])
        #     probs = F.softmax(logits / temperature, dim=-1)
        #     b, l, v = probs.shape
        #     cur_tokens = torch.multinomial(probs.view(-1, v), num_samples=1).view(b, l)

        # log_probs = F.log_softmax(logits.to(torch.float64), dim=-1)
        # cur_scores = torch.gather(log_probs, dim=-1, index=cur_tokens.unsqueeze(-1)).squeeze(-1)

        # print(cur_scores)
        # print(cur_tokens)
        # assert 0
        num_keep = int(total_gen_len * (t / sample_steps))
        
        new_x = output_tokens.clone()
        for b in range(output_tokens.shape[0]):
            temp_scores = output_scores[b].clone()
            temp_scores[~maskable_mask[b]] = -float('inf')
            _, topk_idx = torch.topk(temp_scores, k=num_keep, largest=True)
            # 3. 构造新的序列状态
            # 先默认全填为 MASK
            new_x_b = torch.where(~maskable_mask[b], output_tokens[b], tokenizer.mask_token_id)
            # 被选中的 Top-K 位置填充采样出来的 cur_tokens
            new_x_b[topk_idx] = cur_tokens[b, topk_idx]
            new_x[b] = new_x_b
            
            # 4. 更新持久化的 output_scores，供显示或后续复杂逻辑使用
            output_scores[b, topk_idx] = temp_scores[topk_idx]
            mask_idx = (new_x_b == tokenizer.mask_token_id)
            output_scores[b, mask_idx] = -1e9
        output_tokens = new_x
        # print(x[0].shape)
        # print(x[0])
        print(f"Step {t}/{sample_steps}: {tokenizer.decode(output_tokens[0], skip_special_tokens=False)}")
    return output_tokens


@torch.no_grad()
def dlm_generation_v5(
    dlm, 
    tokenizer,
    seq_len=200, 
    sample_steps=200,
    temperature=1.0,
    verbose=True,
):
    # 重置随机数状态以确保可重复性
    # reset_rng_state()
    
    output_tokens = torch.full((1, seq_len + 2), tokenizer.mask_token_id).to('cuda')
    output_tokens[:, 0] = tokenizer.cls_token_id
    output_tokens[:, -1] = tokenizer.eos_token_id

    output_scores = torch.full_like(output_tokens, -1e9, dtype=torch.bfloat16)
    maskable_mask = torch.zeros_like(output_tokens, dtype=torch.bool)
    maskable_mask[:, 1:-1] = True 

    # 记录哪些位置是mask（xt_neq_x0）
    xt_neq_x0 = maskable_mask.clone()

    for t in range(1, sample_steps + 1):
        logits = dlm.forward(output_tokens).logits

        logits[..., tokenizer.mask_token_id] = -math.inf
        logits[..., tokenizer._token_to_id["X"]] = -math.inf
        logits[..., tokenizer.pad_token_id] = -math.inf
        logits[..., tokenizer.cls_token_id] = -math.inf
        logits[..., tokenizer.eos_token_id] = -math.inf
        
        if temperature == 0:
            cur_tokens = torch.argmax(logits, dim=-1)
            cur_scores = torch.gather(logits.log_softmax(dim=-1), dim=-1, index=cur_tokens.unsqueeze(-1)).squeeze(-1)
        elif temperature > 0:
            probs = F.softmax(logits / temperature, dim=-1)
            b, l, v = probs.shape
            cur_tokens = torch.multinomial(probs.view(-1, v), num_samples=1).view(b, l)
            cur_scores = torch.gather(probs.log(), dim=-1, index=cur_tokens.unsqueeze(-1)).squeeze(-1)
        else:
            noise_scale = 1.0
            cur_tokens, cur_scores = stochastic_sample_from_categorical(
                logits, temperature=0.0, noise_scale=noise_scale
            )
        
        rate = 1 - t / sample_steps
        cutoff_len = (maskable_mask.sum(1, keepdim=True).type_as(cur_scores) * rate).long()
        _scores_for_topk = cur_scores.masked_fill(~maskable_mask, 1000.0)
        
        # 选择最低的cutoff_len个位置（这些位置要mask）
        new_x = output_tokens.clone()
        for b in range(output_tokens.shape[0]):
            _, lowest_k_idx = torch.topk(_scores_for_topk[b], k=cutoff_len[b].item(), largest=False)
            lowest_k_mask = torch.zeros_like(_scores_for_topk[b], dtype=torch.bool)
            lowest_k_mask[lowest_k_idx] = True
            
            # 与_reparam_decoding的逻辑保持一致
            # not_v1_t = lowest_k_mask (uncond模式)
            not_v1_t = lowest_k_mask
            not_v2_t = lowest_k_mask
            
            # masked_to_noise: 需要mask的位置
            masked_to_noise = (~xt_neq_x0[b] & not_v1_t) | (xt_neq_x0[b] & not_v2_t)
            new_x[b, masked_to_noise] = tokenizer.mask_token_id
            output_scores[b, masked_to_noise] = -math.inf
            
            # masked_to_x0: 需要更新为cur_tokens的位置
            masked_to_x0 = xt_neq_x0[b] & ~not_v2_t
            new_x[b, masked_to_x0] = cur_tokens[b, masked_to_x0]
            output_scores[b, masked_to_x0] = cur_scores[b, masked_to_x0]
            
            # 更新xt_neq_x0（记录哪些位置是mask）
            xt_neq_x0[b] = lowest_k_mask
        
        output_tokens = new_x
    
    return output_tokens

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("airkingbd/dplm_150m")
    dlm = AutoModelForCausalLM.from_pretrained(
        "ckpts/cache/difflm/converted_checkpoints/01-1224-dlm_training_ur50_unconditional_1B/ckptstep_307200/hf", 
        torch_dtype="auto", trust_remote_code=True, attn_implementation="flash_attention_2").to('cuda')
    dlm.eval()
    # for dlm generation
    output_tokens = dlm_generation_v5(dlm, tokenizer, temperature=-1)
    seqs = tokenizer.batch_decode(output_tokens, skip_special_tokens=False)
    print(seqs)
    seqs = "".join(seqs[0].split(" "))
    print(seqs)
