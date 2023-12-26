import torch
import torch.nn as nn
import tqdm
import gc
import functools
from collections import defaultdict

from transformers.models.bloom.modeling_bloom import BloomForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM

import random
from transformers.models.bloom.modeling_bloom import BloomBlock, BloomGelu
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.activations import GELUActivation
from .auto_scale import auto_scale_block, get_act_scale
from .auto_clip import auto_clip_block, apply_clip
from .pre_quant import get_blocks, move_embed, get_named_linears
from ..utils.module import get_op_name

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
INT_MAPPING = {"nf4": NF4_BIT, "fp4": FP4_BNB_BIT, "fp4_e2m1_bnb": FP4_BNB_BIT, "fp4_e2m1": FP4_E2M1_BIT}


@torch.no_grad()
def run_daq(
        model, enc,
        w_bit, q_config,
        n_samples=512, seqlen=512,
        auto_scale=True, mse_range=True,
        # some configs for ablation study
        calib_data="pileval",
        hyper_parameters={}
):
    from ..utils.calib_data import get_calib_dataset
    from ..utils.module import append_str_prefix, get_op_name

    if "bigcode" in str(model.__class__).lower():
        # otherwise attention_mask will always be on cpu.
        model.transformer.bias = model.transformer.bias.to("cuda")
    layers = get_blocks(model)

    samples = get_calib_dataset(
        data=calib_data, tokenizer=enc, n_samples=n_samples, block_size=seqlen)
    samples = torch.cat(samples, dim=0)

    inps = []
    layer_kwargs = {}

    layers[0] = layers[0].cuda()
    move_embed(model, "cuda")

    # get input and kwargs to layer 0
    # with_kwargs is only supported in PyTorch 2.0
    # use this Catcher hack for now
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            layer_kwargs.update(kwargs)
            raise ValueError  # early exit to break later inference

    # patch layer 0 to catch input and kwargs
    layers[0] = Catcher(layers[0])
    try:
        model(samples.to(next(model.parameters()).device))
    except ValueError:  # work with early exit
        pass
    del samples
    layers[0] = layers[0].module  # restore
    inps = inps[0]

    layers[0] = layers[0].cpu()
    move_embed(model, "cpu")

    gc.collect()
    torch.cuda.empty_cache()

    awq_results = {
        "scale": [],
        "clip": [],
    }

    # solve layer by layer
    for i in tqdm.tqdm(range(len(layers)), desc="Running AWQ..."):
        layer = layers[i]
        layer = layer.cuda()
        named_linears = get_named_linears(layer)

        # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)

        input_feat = defaultdict(list)
        handles = []
        for name in named_linears:
            handles.append(named_linears[name].register_forward_hook(
                functools.partial(cache_input_hook, name=name,
                                  feat_dict=input_feat)))
        inps = inps.to(next(layer.parameters()).device)  # in case multi-gpu
        # get output as next layer's input
        inps = layer(inps, **layer_kwargs)[0]
        for h in handles:
            h.remove()
        # now solve for scaling and clipping
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

        # Clear GPU memory
        torch.cuda.empty_cache()

        if auto_scale:  # if it applies, we should also modify the input_feat with scales
            scales_list = daq_auto_scale_block(
                layer, layer_kwargs,
                w_bit=w_bit, q_config=q_config,
                input_feat=input_feat, hyper_parameters=hyper_parameters
            )
            # apply_scale(layer, scales_list, input_feat_dict=input_feat)
            # apply_scale(layers[i], scales_list, input_feat_dict=input_feat)
            # append prefix to make names global
            # awq_results["scale"] += append_str_prefix(scales_list, get_op_name(model, layer) + ".")

        # Clear GPU memory
        torch.cuda.empty_cache()

        # if mse_range:
        #     clip_list = auto_clip_block(layer,
        #                                 w_bit=w_bit, q_config=q_config,
        #                                 input_feat=input_feat, )
        #     apply_clip(layer, clip_list)
        #     # append prefix to make names global
        #     awq_results["clip"] += append_str_prefix(clip_list, get_op_name(model, layer) + ".")

        layer = layer.cpu()
        # Haotian: check activation replacement
        del input_feat
        gc.collect()
        torch.cuda.empty_cache()

    return awq_results


@torch.no_grad()
def daq_auto_scale_block(module, module_kwargs,
                         w_bit, q_config,
                         input_feat, hyper_parameters):
    from .quantizer import pseudo_quantize_tensor
    # firstly, get the weight quantize function
    if w_bit is not None:
        def w_quantize_func(p):
            return pseudo_quantize_tensor(
                p, n_bit=w_bit, **q_config,
            ).detach()
    else:
        def w_quantize_func(p):
            return p

    if "use_cache" in module_kwargs:
        module_kwargs.pop("use_cache")

    # find the best scale ratio
    @torch.no_grad()
    def _search_module_scale(block, linears2scale: list, raw_data, kwargs={}):
        # w: co, ci
        # x: n, ci
        raw_data = raw_data.to(next(block.parameters()).device)
        # with torch.no_grad():
        #     org_out = block(x, **kwargs)
        #     if isinstance(org_out, tuple):
        #         org_out = org_out[0]

        data_type = hyper_parameters['data_types']
        bins = hyper_parameters['bins']
        epsilon_org = hyper_parameters['epsilon']
        alpha_org = hyper_parameters['alpha']
        num_epoch = hyper_parameters['num_epoch']
        num_iter = hyper_parameters['num_iter']
        group = hyper_parameters['group']
        allow_data = FLOAT_MAPPING[data_type]
        q_max = allow_data[-2]
        real_max = allow_data[-1]
        org_sd = {k: v.cpu() for k, v in block.state_dict().items()}
        qkv_sd = org_sd.copy()
        flag = 0
        random.seed(2024)
        for fc in linears2scale:
            print(str(fc))
            x = raw_data
            fc_w = fc.weight.data
            feature = fc_w.shape[1]
            if group > 0:
                fc_w = fc_w.reshape(-1, group)
                # bsz, seq, feature = x.shape
                x = x.reshape(-1, group)
            # Step1 初始化scale
            max_val_tensor, _ = torch.max(fc_w, dim=1)
            min_val_tensor, _ = torch.min(fc_w, dim=1)
            histogram = torch.zeros((fc_w.shape[0], bins)).cuda()
            for i, channel in enumerate(fc_w):
                histogram[i] = torch.histc(channel.float(), bins=bins)
            max_bin_index = torch.argmax(histogram, dim=1)
            bin_width = (max_val_tensor - min_val_tensor) / bins
            # density = min_val_tensor + (max_bin_index + .5) * bin_width
            # k = torch.maximum(max_val_tensor - density, density - min_val_tensor)
            # scale = k / q_max
            # zp = -density / scale
            scale = (max_val_tensor - min_val_tensor) / (allow_data[-1] - allow_data[0])
            zp = allow_data[-1] - max_val_tensor / scale
            rate = 5
            shp = raw_data.shape[0]

            # Step2 调优scale 和 weight
            org_out = torch.matmul(x, fc_w.t())
            for i in range(num_epoch):
                # if (i+1) * 64 % shp != 0:
                #     x = raw_data[i * 64 % shp: (i+1) * 64 % shp]
                # else:
                #     x = raw_data[i * 64 % shp:]

                flag += 1
                min_epsilon = epsilon_org / rate
                max_epsilon = epsilon_org * rate
                min_alpha = alpha_org / rate
                max_alpha = alpha_org * rate
                for j in range(num_iter):
                    if j % 2:
                        min_epsilon = 0.9 * min_epsilon
                        min_alpha = 0.9 * min_alpha
                        epsilon = min_epsilon
                        alpha = min_alpha
                    else:
                        max_epsilon = 1.1 * max_epsilon
                        max_alpha = 1.1 * max_alpha
                        epsilon = max_epsilon
                        alpha = max_alpha
                    mid_loss, mid_o = get_loss_by_s_and_zp(fc, zp, scale, data_type, org_out, x, group, **kwargs)
                    left_loss, left_o = get_loss_by_s_and_zp(fc, zp - epsilon, scale, data_type, org_out, x, group, **kwargs)
                    right_loss, right_o = get_loss_by_s_and_zp(fc, zp + epsilon, scale, data_type, org_out, x, group,**kwargs)
                    gradient_sign = torch.sign(right_loss - left_loss)
                    gradient_sign = torch.where(torch.logical_or(left_loss < mid_loss, right_loss < mid_loss),
                                                gradient_sign, 0)
                    # gradient_sign = torch.where(
                    #     torch.logical_and(max_val_tensor / scale + zp + epsilon < allow_data[-1] * 1.2,
                    #                      min_val_tensor / scale + zp - epsilon > allow_data[0] * 1.2),
                    #     gradient_sign, 0)
                    print(mid_loss.sum(), alpha, epsilon)
                    if gradient_sign.abs().sum() == 0:
                        break
                    flag = 0
                    zp = zp - alpha * gradient_sign
                min_epsilon = epsilon_org / rate
                max_epsilon = epsilon_org * rate
                min_alpha = alpha_org / rate
                max_alpha = alpha_org * rate
                for _ in range(num_iter):
                    if j % 2:
                        min_epsilon = 0.9 * min_epsilon
                        min_alpha = 0.9 * min_alpha
                        epsilon = min_epsilon
                        alpha = min_alpha
                    else:
                        max_epsilon = 1.1 * max_epsilon
                        max_alpha = 1.1 * max_alpha
                        epsilon = max_epsilon
                        alpha = max_alpha
                    mid_loss, _ = get_loss_by_s_and_zp(fc, zp, scale, data_type, org_out, x, group, **kwargs)
                    left_scale = scale * (1 - epsilon)
                    # left_scale = torch.where(torch.logical_and(left_scale > 0, min_val_tensor / left_scale + zp < allow_data[-1] * 1.2), left_scale, scale)
                    left_loss, _ = get_loss_by_s_and_zp(fc, zp, left_scale, data_type, org_out, x, group, **kwargs)
                    right_scale = scale * (1 + epsilon)
                    right_loss, _ = get_loss_by_s_and_zp(fc, zp, right_scale, data_type, org_out, x, group, **kwargs)
                    gradient_sign = torch.sign(right_loss - left_loss)
                    gradient_sign = torch.where(torch.logical_or(left_loss < mid_loss, right_loss < mid_loss),
                                                gradient_sign, 0)
                    # gradient_sign = torch.where(
                    #     torch.logical_and(max_val_tensor / (scale - epsilon) + zp < allow_data[-1] * 1.2,
                    #                      min_val_tensor / (scale + epsilon) + zp > allow_data[0] * 1.2),
                    #     gradient_sign, 0)
                    print(mid_loss.sum(), alpha, epsilon)
                    if gradient_sign.abs().sum() == 0:
                        break
                    flag = 0
                    new_scale = torch.where(left_loss < right_loss, left_scale, right_scale)
                    new_scale = torch.where(gradient_sign != 0, new_scale, scale)
                    scale = new_scale
                print(mid_loss.sum())
                if flag > 5:
                    print(flag)
                    flag = 0
                    break
            old_out = fc(raw_data)
            fc.weight.data = _daq_qdq_4bit(fc.weight.data, scale, zp, data_type, group).reshape(-1, feature)
            new_out = fc(raw_data)
            print('real:', (old_out-new_out).float().pow(2).mean(dim=0).mean(dim=0).sum())
            best_scales = scale.view(-1)
            best_zp = zp.view(-1)

        assert torch.isnan(best_scales).sum() == 0, best_scales
        assert torch.isnan(best_zp).sum() == 0, best_zp
        return best_scales.detach(), best_zp.detach()

    def _auto_get_scale(prev_op, layers, inp, module2inspect=None, kwargs={}):
        # module2inspect: if given, we will check the output diff of this module instead of layers
        if module2inspect is None:
            assert len(layers) == 1
            module2inspect = layers[0]

        scales, zp = _search_module_scale(module2inspect, layers, inp, kwargs)
        scales = scales.detach().cpu()
        zp = zp.detach().cpu()
        # prev_op_name, [layer_name], scale
        return (get_op_name(module, prev_op), tuple([get_op_name(module, m) for m in layers]), scales, zp)

    scales_list = []  # return the searched scales

    if isinstance(module, OPTDecoderLayer):
        # attention input
        scales_list.append(_auto_get_scale(
            prev_op=module.self_attn_layer_norm,
            layers=[module.self_attn.q_proj,
                    module.self_attn.k_proj, module.self_attn.v_proj],
            inp=input_feat['self_attn.q_proj'],
            module2inspect=module.self_attn, kwargs=module_kwargs,
        ))
        # attn out
        scales_list.append(_auto_get_scale(
            prev_op=module.self_attn.v_proj,
            layers=[module.self_attn.out_proj],
            inp=input_feat['self_attn.out_proj'],
        ))
        # fc1
        scales_list.append(_auto_get_scale(
            prev_op=module.final_layer_norm,
            layers=[module.fc1],
            inp=input_feat['fc1'],
        ))
        # fc2
        scales_list.append(_auto_get_scale(
            prev_op=module.fc1,
            layers=[module.fc2],
            inp=input_feat['fc2'],
        ))

    elif isinstance(module, LlamaDecoderLayer):
        # attention input
        scales_list.append(_auto_get_scale(
            prev_op=module.input_layernorm,
            layers=[module.self_attn.q_proj,
                    module.self_attn.k_proj, module.self_attn.v_proj],
            inp=input_feat['self_attn.q_proj'],
            module2inspect=module.self_attn, kwargs=module_kwargs,
        ))
        # attn out
        # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            scales_list.append(_auto_get_scale(
                prev_op=module.self_attn.v_proj,
                layers=[module.self_attn.o_proj],
                inp=input_feat['self_attn.o_proj'],
            ))
        # fc1
        scales_list.append(_auto_get_scale(
            prev_op=module.post_attention_layernorm,
            layers=[module.mlp.gate_proj, module.mlp.up_proj],
            inp=input_feat['mlp.gate_proj'],
            module2inspect=module.mlp,
        ))
        # fc2
        scales_list.append(_auto_get_scale(
            prev_op=module.mlp.up_proj,
            layers=[module.mlp.down_proj],
            inp=input_feat['mlp.down_proj'],
        ))

    elif isinstance(module, BloomBlock):
        # attention input
        scales_list.append(_auto_get_scale(
            prev_op=module.input_layernorm,
            layers=[module.self_attention.query_key_value],
            inp=input_feat['self_attention.query_key_value'],
            module2inspect=module, kwargs=module_kwargs,
        ))
        # attn out
        # Please refer to https://github.com/mit-han-lab/llm-awq/issues/2#issuecomment-1606297469
        """
        scales_list.append(_auto_get_scale(
            prev_op=module.self_attention.query_key_value,
            layers=[module.self_attention.dense],
            inp=input_feat['self_attention.dense'],
        ))
        """
        # fc1
        scales_list.append(_auto_get_scale(
            prev_op=module.post_attention_layernorm,
            layers=[module.mlp.dense_h_to_4h],
            inp=input_feat['mlp.dense_h_to_4h'],
            module2inspect=module, kwargs=module_kwargs,
        ))
        # fc2
        scales_list.append(_auto_get_scale(
            prev_op=module.mlp.gelu_impl,
            layers=[module.mlp.dense_4h_to_h],
            inp=input_feat['mlp.dense_4h_to_h'],
        ))
    elif "mpt" in str(module.__class__).lower():
        # attention input
        scales_list.append(_auto_get_scale(
            prev_op=module.norm_1,
            layers=[module.attn.Wqkv],
            inp=input_feat['attn.Wqkv'],
            module2inspect=module.attn,
            kwargs=module_kwargs,
        ))

        # attn out
        scales_list.append(_auto_get_scale(
            prev_op=module.attn.Wqkv,
            layers=[module.attn.out_proj],
            inp=input_feat['attn.out_proj'],
        ))
        # fc1
        scales_list.append(_auto_get_scale(
            prev_op=module.norm_2,
            layers=[module.ffn.up_proj],
            inp=input_feat['ffn.up_proj'],
            module2inspect=module.ffn,
        ))
        # fc2
        scales_list.append(_auto_get_scale(
            prev_op=module.ffn.act,
            layers=[module.ffn.down_proj],
            inp=input_feat['ffn.down_proj'],
        ))

    elif "falcon" in str(module.__class__).lower():
        # attn out
        # Haotian: TBD: need to handle repeated scales for MQ
        """ 
        scales_list.append(_auto_get_scale(
            prev_op=module.self_attention.query_key_value,
            layers=[module.self_attention.dense],
            inp=input_feat['self_attention.dense'],
        ))
        """
        # fc1, as long as it is scaled, everything is screwed up
        if "falcon-7b" in str(module.__class__).lower():
            scales_list.append(_auto_get_scale(
                prev_op=module.input_layernorm,
                layers=[module.mlp.dense_h_to_4h, module.self_attention.query_key_value],
                inp=input_feat['self_attention.query_key_value'],
                module2inspect=module,
                kwargs=module_kwargs,
            ))
        elif "falcon-40b" in str(module.__class__).lower():
            scales_list.append(_auto_get_scale(
                prev_op=module.ln_attn,
                layers=[module.self_attention.query_key_value],
                inp=input_feat['self_attention.query_key_value'],
                module2inspect=module,
                kwargs=module_kwargs,
            ))
            scales_list.append(_auto_get_scale(
                prev_op=module.ln_mlp,
                layers=[module.mlp.dense_h_to_4h],
                inp=input_feat['mlp.dense_h_to_4h'],
                module2inspect=module,
                kwargs=module_kwargs,
            ))
        else:
            raise NotImplementedError(
                "Unknown Falcon architecture, currently only falcon-7b and falcon-40b are supported")
        # fc2
        scales_list.append(_auto_get_scale(
            prev_op=module.mlp.act,
            layers=[module.mlp.dense_4h_to_h],
            inp=input_feat['mlp.dense_4h_to_h'],
        ))
    elif "bigcode" in str(module.__class__).lower():
        scales_list.append(_auto_get_scale(
            prev_op=module.ln_1,
            layers=[module.attn.c_attn],
            inp=input_feat['attn.c_attn'],
            module2inspect=module.attn,
            kwargs=module_kwargs,
        ))
        # fc1
        scales_list.append(_auto_get_scale(
            prev_op=module.ln_2,
            layers=[module.mlp.c_fc],
            inp=input_feat['mlp.c_fc'],
            module2inspect=module.mlp,
        ))
        # fc2
        scales_list.append(_auto_get_scale(
            prev_op=module.mlp.act,
            layers=[module.mlp.c_proj],
            inp=input_feat['mlp.c_proj'],
        ))
    elif "neox" in str(module.__class__).lower():
        scales_list.append(_auto_get_scale(
            prev_op=module.input_layernorm,
            layers=[module.attention.query_key_value],
            inp=input_feat['attention.query_key_value'],
            module2inspect=module.attention,
            kwargs=module_kwargs,
        ))
        # fc1
        scales_list.append(_auto_get_scale(
            prev_op=module.post_attention_layernorm,
            layers=[module.mlp.dense_h_to_4h],
            inp=input_feat['mlp.dense_h_to_4h'],
            module2inspect=module.mlp,
        ))
        # fc2
        scales_list.append(_auto_get_scale(
            prev_op=module.mlp.act,
            layers=[module.mlp.dense_4h_to_h],
            inp=input_feat['mlp.dense_4h_to_h'],
        ))
    else:
        raise NotImplementedError(f"{type(module)} not supported yet!")

    return scales_list


def get_loss_by_s_and_zp(module, zp, s, data_type, org_out, input_data, group=-1,**kwargs):
    org_w = module.weight.data
    now_w = _daq_qdq_4bit(org_w, s, zp, data_type, group)
    # module.weight.data = now_w
    out = torch.matmul(input_data, now_w.t())
    loss = (out - org_out).float().pow(2)
    while len(loss.shape) > 1:
        loss = loss.mean(dim=0)
    # module.weight.data = org_w
    return loss, out


def init_zp_scale(weight, q_max, bins):
    max_val_tensor, _ = torch.max(weight, dim=1)
    min_val_tensor, _ = torch.min(weight, dim=1)
    histogram = torch.zeros((weight.shape[0], bins)).cuda()
    for i, channel in enumerate(weight):
        histogram[i] = torch.histc(channel.float(), bins=bins)
    max_bin_index = torch.argmax(histogram, dim=1)
    bin_width = (max_val_tensor - min_val_tensor) / bins
    density = min_val_tensor + (max_bin_index + .5) * bin_width
    k = torch.maximum(max_val_tensor - density, density - min_val_tensor)
    scale = k / q_max
    zeropoint = -density / scale
    return scale, zeropoint


def _daq_qdq_4bit(weight, scale, zp, data_type, group=-1):
    assert data_type in FLOAT_MAPPING, "unexpected data type."
    if group > 0:
        weight = weight.reshape(-1, group)
    allow_data = FLOAT_MAPPING[data_type]
    # scale = scale.copy()
    scale = scale.unsqueeze(dim=-1)
    zp = zp.unsqueeze(dim=-1)
    tensor = weight / scale + zp
    mid_data = [(allow_data[i] + allow_data[i + 1]) / 2 for i in range(len(allow_data) - 1)]
    q_tensor = torch.zeros_like(tensor)
    for i in range(len(allow_data)):
        data = allow_data[i]
        if i == 0:
            q_tensor += torch.where(tensor <= mid_data[i], data, 0)
        elif i == len(allow_data) - 1:
            q_tensor += torch.where(tensor > mid_data[i - 1], data, 0)
        else:
            q_tensor += torch.where((mid_data[i - 1] < tensor) & (tensor <= mid_data[i]), data, 0)
    return ((q_tensor - zp) * scale).half()
