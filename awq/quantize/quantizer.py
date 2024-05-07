import torch
import torch.nn as nn
from tqdm import tqdm
import gc
from .qmodule import ScaledActivation
from ..utils.module import set_op_by_name

from transformers.models.bloom.modeling_bloom import BloomBlock

NF4 = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
]
FP4_BNB = [-12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -0.0625, 0, 0.0625, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
FP4_E2M1 = [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.0625, 0, 0.0625, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

# the order is the same as float list, bit value range is [-7, 7]
# 1111 = -1, 1110 = -2, 1101= -3, ...

NF4_BIT = [7, 1, 2, 3, 4, 5, 6, 0, -8, -7, -6, -5, -4, -3, -2, -1]
FP4_BNB_BIT = [-5, -6, -3, -4, -1, -2, -7, 0, 1, 6, 7, 4, 5, 2, 3]
FP4_E2M1_BIT = [-1, -2, -3, -4, -5, -6, -7, 0, 1, 2, 3, 4, 5, 6, 7]

FLOAT_MAPPING = {"nf4": NF4, "fp4": FP4_BNB, "fp4_e2m1_bnb": FP4_BNB, "fp4_e2m1": FP4_E2M1}


EMBEDDING_KEYWORDS = ["embed"]
LM_HEAD_KEYWORDS = ["lm_head", "embed_out", "output"]


def scale_activations(module):
    param = next(module.parameters())
    dtype = param.dtype
    device = param.device
    if isinstance(module, BloomBlock):
        if isinstance(module.mlp.gelu_impl, ScaledActivation):
            return
        c = module.mlp.dense_h_to_4h.out_features
        act = ScaledActivation(
            module.mlp.gelu_impl, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.gelu_impl", act)
    elif "mptblock" in str(module.__class__.__name__).lower():
        if isinstance(module.ffn.act, ScaledActivation):
            return
        c = module.ffn.up_proj.out_features
        act = ScaledActivation(
            module.ffn.act, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "ffn.act", act)
    elif "falcon" in str(module.__class__).lower():
        if isinstance(module.mlp.act, ScaledActivation):
            return
        c = module.mlp.dense_h_to_4h.out_features
        act = ScaledActivation(
            module.mlp.act, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.act", act)
    elif "bigcode" in str(module.__class__).lower():
        if isinstance(module.mlp.act, ScaledActivation):
            return
        c = module.mlp.c_proj.out_features
        act = ScaledActivation(
            module.mlp.act, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.act", act)
    elif "neox" in str(module.__class__).lower():
        if isinstance(module.mlp.act, ScaledActivation):
            return
        c = module.mlp.dense_h_to_4h.out_features
        act = ScaledActivation(
            module.mlp.act, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.act", act)


def quantize_nf(x, q_group_size, code, get_scale_zp):
    org_shape = x.shape
    if q_group_size > 0:
        x = x.reshape(-1, q_group_size)
    else:
        x = x.squeeze()
    max_val_tensor, _ = torch.max(x, dim=1)
    min_val_tensor, _ = torch.min(x, dim=1)
    scale = (max_val_tensor - min_val_tensor) / (code[-1] - code[0])
    zp = code[-1] - max_val_tensor / scale
    scale = scale.unsqueeze(dim=-1)
    zp = zp.unsqueeze(dim=-1)
    q = quant_asym(code, scale, x, zp)
    q = q.reshape(org_shape)
    if get_scale_zp:
        return q, scale.view(q.shape[0], -1), zp.view(q.shape[0], -1)
    else:
        return q


def quant_asym(code, scale, x, zp):
    q = x / scale + zp
    # q = q.reshape(-1, 1)
    # distance = torch.abs(q - code)
    # idx = torch.argmin(distance, dim=-1)
    # q = torch.gather(code, -1, idx)
    mid_data = [(code[i] + code[i + 1]) / 2 for i in range(len(code) - 1)]
    q_tensor = torch.zeros_like(q)
    for i in range(len(code)):
        data = code[i]
        if i == 0:
            q_tensor += torch.where(q <= mid_data[i], data, 0)
        elif i == len(code) - 1:
            q_tensor += torch.where(q > mid_data[i - 1], data, 0)
        else:
            q_tensor += torch.where((mid_data[i - 1] < q) & (q <= mid_data[i]), data, 0)
    q = q_tensor
    q = (q - zp) * scale
    return q.half()


def quantize_nf_sym(x, q_group_size, code, get_scale_zp):
    org_shape = x.shape
    if q_group_size > 0:
        x = x.reshape(-1, q_group_size)
    else:
        x = x.squeeze()
    max_val_tensor, _ = torch.max(x.abs(), dim=1)
    scale = max_val_tensor / code[-1]
    scale = scale.unsqueeze(dim=-1)
    q = x / scale
    mid_data = [(code[i] + code[i + 1]) / 2 for i in range(len(code) - 1)]
    q_tensor = torch.zeros_like(q)
    for i in range(len(code)):
        data = code[i]
        if i == 0:
            q_tensor += torch.where(q <= mid_data[i], data, 0)
        elif i == len(code) - 1:
            q_tensor += torch.where(q > mid_data[i - 1], data, 0)
        else:
            q_tensor += torch.where((mid_data[i - 1] < q) & (q <= mid_data[i]), data, 0)
    q = q_tensor
    q = q * scale
    q = q.reshape(org_shape).half()
    if get_scale_zp:
        return q, scale.view(q.shape[0], -1)
    else:
        return q

def nf4_quantize(original_func):
    def wrapper(w, n_bit=8, zero_point=True, q_group_size=-1, inplace=False, get_scale_zp=False, data_type='int'):
        if data_type == 'int':
            return original_func(w, n_bit, zero_point, q_group_size, inplace, get_scale_zp, data_type)
        else:
            if data_type not in FLOAT_MAPPING.keys():
                raise ValueError("Wrong data type")
            code = torch.tensor(FLOAT_MAPPING[data_type]).to(w.device)
            if zero_point:
                return quantize_nf(w, q_group_size, code, get_scale_zp)
            else:
                return quantize_nf_sym(w, q_group_size, code, get_scale_zp)
    return wrapper


# core quantization method (simulated quantization)
@nf4_quantize  # the decorator hack the original logic, pls refer to function nf4_quantize
def pseudo_quantize_tensor(
    w, n_bit=8, zero_point=True, q_group_size=-1, inplace=False, get_scale_zp=False, data_type='int'
):
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    # print(w.shape)
    w = w.squeeze()
    assert w.dim() == 2
    if zero_point:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2**n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        # zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
        zeros = - min_val / scales
    else:  # we actually never used this
        # assert min_val is None
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (n_bit - 1) - 1
        min_int = -(2 ** (n_bit - 1))
        scales = max_val / max_int
        zeros = 0

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    if inplace:
        (
            (w.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int).sub_(zeros)
        ).mul_(scales)
    else:
        w = (
            torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
        ) * scales
    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)

    if get_scale_zp:
        return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
    else:
        return w


@torch.no_grad()
def pseudo_quantize_model_weight(
    model,
    w_bit,
    q_config,
):
    from .pre_quant import get_blocks, get_named_linears

    layers = get_blocks(model)
    for i in tqdm(range(len(layers)), desc="pseudo weight quantization..."):
        named_linears = get_named_linears(layers[i])
        for n, m in named_linears.items():
            m.cuda()
            m.weight.data = pseudo_quantize_tensor(
                m.weight.data, n_bit=w_bit, **q_config
            )
            m.cpu()


@torch.no_grad()
def real_quantize_model_weight(model, w_bit, q_config, init_only=False):
    from .qmodule import WQLinear
    from .pre_quant import get_blocks, get_named_linears

    assert q_config["zero_point"], "We only support zero_point quantization now."

    layers = get_blocks(model)
    for i in tqdm(
        range(len(layers)),
        desc="real weight quantization..." + ("(init only)" if init_only else ""),
    ):
        layer = layers[i]
        named_linears = get_named_linears(layer)
        scale_activations(layer)

        for name, module in named_linears.items():
            if init_only:
                q_linear = WQLinear.from_linear(
                    module, w_bit, q_config["q_group_size"], True
                )
                q_linear.to(next(layer.parameters()).device)
                set_op_by_name(layer, name, q_linear)
            else:
                module.cuda()
                module.weight.data, scales, zeros = pseudo_quantize_tensor(
                    module.weight.data, n_bit=w_bit, get_scale_zp=True, **q_config
                )
                # scales = scales.t().contiguous()
                # zeros = zeros.t().contiguous()
                q_linear = WQLinear.from_linear(
                    module, w_bit, q_config["q_group_size"], False, scales, zeros
                )
                module.cpu()
                q_linear.to(next(layer.parameters()).device)
                set_op_by_name(layer, name, q_linear)
                torch.cuda.empty_cache()
                gc.collect()

    torch.cuda.empty_cache()
    gc.collect()
