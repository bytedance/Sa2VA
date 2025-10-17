import warnings
from typing import Optional, Tuple

import torch
from eve.model.language_model.qwen2.modeling_qwen2 import Qwen2Model, Qwen2Attention, Qwen2SdpaAttention, apply_rotary_pos_emb, repeat_kv

try:
    from flash_attn.flash_attn_interface import \
        flash_attn_unpadded_qkvpacked_func
except ImportError:
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func as flash_attn_unpadded_qkvpacked_func

from flash_attn.bert_padding import pad_input, unpad_input


# ---- Haiwen Diao ---- #
def moe_fuction(hidden_states, visual_token_mask, raw_layers, moe_layers, training):
    hidden_states = raw_layers(hidden_states) * (1. - visual_token_mask) \
                    + moe_layers(hidden_states) * visual_token_mask
    return hidden_states


# ---- Haiwen Diao ---- #
# def moe_fuction(hidden_states, visual_token_mask, raw_layers, moe_layers, training):
#     if training:
#         hidden_states = raw_layers(hidden_states) * (1. - visual_token_mask) \
#                         + moe_layers(hidden_states) * visual_token_mask
#     else:
#         dim = hidden_states.shape[-1]
#         visual_token_mask = visual_token_mask.repeat(1, 1, dim).bool()
#         non_visual_token_mask = ~visual_token_mask
#         if visual_token_mask.any():
#             hidden_states[visual_token_mask] = moe_layers(hidden_states[visual_token_mask].reshape(-1, dim)).reshape(-1)
#         if (non_visual_token_mask).any(): 
#             hidden_states[non_visual_token_mask] = raw_layers(hidden_states[non_visual_token_mask].reshape(-1, dim)).reshape(-1)
#     return hidden_states


def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    visual_token_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

    assert past_key_value is None
    if output_attentions:
        warnings.warn(
            "Output attentions is not supported for patched `Attention`, returning `None` instead."
        )

    bsz, q_len, _ = hidden_states.size()

    if self.config.add_moe and ('self_attn' in self.config.moe_part) and (hidden_states.shape[1] != 1):
        query_states = moe_fuction(hidden_states, visual_token_mask, self.q_proj, self.moe_q_proj, self.training)
        key_states = moe_fuction(hidden_states, visual_token_mask, self.k_proj, self.moe_k_proj, self.training)
        value_states = moe_fuction(hidden_states, visual_token_mask, self.v_proj, self.moe_v_proj, self.training)
    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = (
        query_states
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    key_states = (
        key_states
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    value_states = (
        value_states
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )  # shape: (b, num_heads, s, head_dim)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    if past_key_value is not None:
        # reuse k, v
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # Transform the data into the format required by flash attention
    qkv = torch.stack([query_states, key_states, value_states], dim=2)
    qkv = qkv.transpose(1, 3)  # shape: [b, s, 3, num_heads, head_dim]
    key_padding_mask = attention_mask

    if key_padding_mask is None:
        qkv = qkv.reshape(-1, 3, self.num_heads, self.head_dim)
        cu_q_lens = torch.arange(
            0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32, device=qkv.device
        )
        max_s = q_len
        output = flash_attn_unpadded_qkvpacked_func(
            qkv, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
        )
        output = output.view(bsz, q_len, -1)
    else:
        qkv = qkv.reshape(bsz, q_len, -1)
        qkv, indices, cu_q_lens, max_s = unpad_input(qkv, key_padding_mask)
        qkv = qkv.view(-1, 3, self.num_heads, self.head_dim)
        output_unpad = flash_attn_unpadded_qkvpacked_func(
            qkv, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
        )
        output_unpad = output_unpad.reshape(-1, self.num_heads * self.head_dim)
        output = pad_input(output_unpad, indices, bsz, q_len)
    
    if self.config.add_moe and ('self_attn' in self.config.moe_part) and (hidden_states.shape[1] != 1):
        output = moe_fuction(output, visual_token_mask, self.o_proj, self.moe_o_proj, self.training)                                                                       
    else:
        output = self.o_proj(output)

    return output, None, past_key_value


# Disable the transformation of the attention mask in Model as the flash attention
# requires the attention mask to be the same as the key_padding_mask
def _prepare_decoder_attention_mask(
    self, attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    # [bsz, seq_len]
    return attention_mask


def flash_attn_replace():
    cuda_major, cuda_minor = torch.cuda.get_device_capability()
    if cuda_major < 8:
        warnings.warn(
            "Flash attention is only supported on A100 or H100 GPU during training due to head dim > 64 backward."
            "ref: https://github.com/HazyResearch/flash-attention/issues/190#issuecomment-1523359593"
        )
    Qwen2Model._prepare_decoder_attention_mask = (
        _prepare_decoder_attention_mask
    )
    Qwen2Attention.forward = forward
