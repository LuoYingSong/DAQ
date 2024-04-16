import torch
import torch.nn as nn
import tqdm
import gc
import functools
from collections import defaultdict
import queue
import threading
import logging
import random
from transformers.models.bloom.modeling_bloom import BloomBlock, BloomGelu
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from .pre_quant import get_blocks, move_embed, get_named_linears
from ..utils.module import get_op_name, get_op_by_name
from ..utils.module import append_str_prefix, get_op_name
import sys

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

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


def init_zp_scale(allow_data, bins, fc_w):
    max_val_tensor, _ = torch.max(fc_w, dim=1)
    min_val_tensor, _ = torch.min(fc_w, dim=1)
    histogram = torch.zeros((fc_w.shape[0], bins)).cuda()

    for i, channel in enumerate(fc_w):
        histogram[i] = torch.histc(channel.float(), bins=bins)
    extended_hist = torch.cat(
        [torch.zeros(histogram.shape[0], 1).cuda(), histogram, torch.zeros(histogram.shape[0], 1).cuda()], dim=1)

    # 初始化左右游标
    mode_index = torch.argmax(histogram, dim=1) + 1
    left_cursor = mode_index.clone()
    right_cursor = mode_index.clone()

    # 初始化累积直方图
    accumulated = extended_hist[torch.arange(histogram.shape[0]), mode_index]
    half_total = histogram.sum(dim=1) / 2

    # 并行搜索
    while True:
        move_mask = accumulated < half_total
        if not move_mask.abs().sum():
            break

        # 同时更新所有左右游标
        left_move = extended_hist[torch.arange(histogram.shape[0]), left_cursor - 1]
        right_move = extended_hist[torch.arange(histogram.shape[0]), right_cursor + 1]

        move_left = (left_move > right_move) & move_mask
        move_right = ~move_left & move_mask

        can_move_left = (left_cursor > 1) & move_mask
        can_move_right = (right_cursor < bins) & move_mask

        move_left = move_left & can_move_left | (~can_move_right & move_mask)
        move_right = move_right & can_move_right | (~can_move_left & move_mask)

        left_cursor = torch.where(move_left, left_cursor - 1, left_cursor)
        right_cursor = torch.where(move_right, right_cursor + 1, right_cursor)

        # 更新累积直方图
        accumulated += torch.where(move_left, left_move, right_move)

    # 计算中间点
    bin_width = (max_val_tensor - min_val_tensor) / bins
    left_val = min_val_tensor + (left_cursor - 1) * bin_width
    right_val = min_val_tensor + right_cursor * bin_width
    density = (left_val + right_val) / 2

    # k = torch.maximum(max_val_tensor - density, density - min_val_tensor)
    # scale = k / allow_data[-1]

    scale = (max_val_tensor - min_val_tensor) / (allow_data[-1] - allow_data[0])
    # zp = - density * scale
    zp = allow_data[-1] - max_val_tensor / scale
    return scale, zp


@torch.no_grad()
def search_module_scale(block, linears2scale: list, raw_data, kwargs={}, hyper_parameters={}):
    # w: co, ci
    # x: n, ci
    raw_data = raw_data.to(next(block.parameters()).device)
    data_type = hyper_parameters['data_types']
    bins = hyper_parameters['bins']
    epsilon_org = hyper_parameters['epsilon']
    alpha_org = hyper_parameters['alpha']
    num_epoch = hyper_parameters['num_epoch']
    num_iter = hyper_parameters['num_iter']
    group = hyper_parameters['group']
    allow_data = FLOAT_MAPPING[data_type]
    flag = 0
    random.seed(2024)
    best_scales_list = []
    best_zp_list = []
    for fc in linears2scale:
        logging.debug("%s", str(fc))
        x = raw_data
        fc_w = fc.weight.data
        logging.info("old x:%s w:%s",x.shape, fc_w.shape)
        if group > 0:
            fc_w = fc_w.reshape(-1, group)
        # if len(x.shape) > 2:
        #     global bsz, seq
        #     bsz, seq, feature = x.shape
        # else:
        #     # x = x.reshape(bsz, seq, -1)
        #     pass
        # x = x.sum(dim=0)
        # Step1 初始化scale
        logging.info("new x:%s w:%s",x.shape, fc_w.shape)
        scale, zp = init_zp_scale(allow_data, bins, fc_w)
        rate = 5

        # Step2 调优scale 和 weight
        org_out = cal_matrix(x, fc.weight.data, group)
        for i in range(num_epoch):
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
                mid_loss = get_loss_by_s_and_zp(fc, zp, scale, data_type, org_out, x, group, **kwargs)
                left_loss = get_loss_by_s_and_zp(fc, zp - epsilon, scale, data_type, org_out, x, group,
                                                 **kwargs)
                right_loss = get_loss_by_s_and_zp(fc, zp + epsilon, scale, data_type, org_out, x, group,
                                                  **kwargs)
                gradient_sign = torch.sign(right_loss - left_loss)
                gradient_sign = torch.where(torch.logical_or(left_loss < mid_loss, right_loss < mid_loss),
                                            gradient_sign, 0)
                # print(1,torch.cuda.memory_allocated() / (1024 * 1024))

                zp = zp - alpha * gradient_sign

                del mid_loss, left_loss, right_loss

                mid_loss = get_loss_by_s_and_zp(fc, zp, scale, data_type, org_out, x, group, **kwargs)
                # print(2,torch.cuda.memory_allocated()/ (1024 * 1024))

                left_scale = scale * (1 - epsilon)
                left_scale_loss = get_loss_by_s_and_zp(fc, zp, left_scale, data_type, org_out, x, group,
                                                       **kwargs)
                # print(3,torch.cuda.memory_allocated()/ (1024 * 1024))

                right_scale = scale * (1 + epsilon)
                right_scale_loss = get_loss_by_s_and_zp(fc, zp, right_scale, data_type, org_out, x, group,
                                                        **kwargs)
                # print(4,torch.cuda.memory_allocated()/ (1024 * 1024))
                gradient_scale_sign = torch.sign(right_scale_loss - left_scale_loss)
                gradient_scale_sign = torch.where(
                    torch.logical_or(left_scale_loss < mid_loss, right_scale_loss < mid_loss),
                    gradient_scale_sign, 0)
                new_scale = torch.where(left_scale_loss < right_scale_loss, left_scale, right_scale)
                new_scale = torch.where(gradient_scale_sign != 0, new_scale, scale)

                logging.debug("mid_loss sum: %s, alpha: %s, epsilon: %s", mid_loss.sum(), alpha, epsilon)

                scale = new_scale
                if gradient_sign.abs().sum() == 0 and gradient_scale_sign.abs().sum() == 0:
                    flag += 1
                    if flag > 5:
                        break
                else:
                    flag = 0
                del mid_loss, left_scale_loss, right_scale_loss, left_scale, right_scale, gradient_scale_sign, gradient_sign

        best_scales = scale.view(-1).detach().cpu()
        best_zp = zp.view(-1).detach().cpu()
        assert torch.isnan(best_scales).sum() == 0, best_scales
        assert torch.isnan(best_zp).sum() == 0, best_zp
        best_scales_list.append(best_scales)
        best_zp_list.append(zp)

    return best_scales_list, best_zp_list


@torch.no_grad()
def run_daq(
        model, enc,
        w_bit, q_config,
        n_samples=512, seqlen=512,
        auto_scale=True, mse_range=True,
        # some configs for ablation study
        calib_data="pileval",
        token_size=None,
        hyper_parameters={}
):
    from ..utils.calib_data import get_calib_dataset

    if "bigcode" in str(model.__class__).lower():
        # otherwise attention_mask will always be on cpu.
        model.transformer.bias = model.transformer.bias.to("cuda")
    layers = get_blocks(model)

    samples = get_calib_dataset(
        data=calib_data, tokenizer=enc, n_samples=n_samples, block_size=seqlen, token_size=token_size)
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
        "daq": []
    }

    # solve layer by layer
    layer_queue = queue.Queue()
    for i in tqdm.tqdm(range(len(layers)), desc="Running DAQ..."):
        layer = layers[i]
        named_linears = get_named_linears(layer)
        layer = layer.cuda()

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
        layer = layer.cpu()
        for h in handles:
            h.remove()
        # now solve for scaling and clipping
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

        # Clear GPU memory
        torch.cuda.empty_cache()

        layer_queue.put((i, awq_results, hyper_parameters, input_feat, layer, layer_kwargs, model, q_config, w_bit))
    threads = []
    for device in [f'cuda:{i}' for i in range(torch.cuda.device_count())]:
        # for i in range(4):
        thread = threading.Thread(target=consumer, args=(layer_queue, device,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()
    return awq_results


def consumer(layer_queue, device):
    while True:
        try:
            # 从队列中获取任务，设置超时以防队列为空时线程永久阻塞
            i, awq_results, hyper_parameters, input_feat, layer, layer_kwargs, model, q_config, w_bit = layer_queue.get(
                timeout=5)
            logging.info(f"Consuming %s layer %s", str(i), str(layer))
            logging.info(f"Consuming layer in device:%s", str(device))
        except queue.Empty:
            # 队列为空，退出线程
            break
        with torch.cuda.device(device):
            layer = layer.to(device)
            for k in input_feat.keys():
                input_feat[k] = input_feat[k].to(device)
            process_one_layer(awq_results, hyper_parameters, input_feat, layer, layer_kwargs, model, q_config, w_bit)


def process_one_layer(awq_results, hyper_parameters, input_feat, layer, layer_kwargs, model, q_config, w_bit):
    scales_list = daq_auto_scale_block(
        layer, layer_kwargs,
        w_bit=w_bit, q_config=q_config,
        input_feat=input_feat, hyper_parameters=hyper_parameters
    )
    daq_apply_scale(layer, scales_list, hyper_parameters['data_types'], input_feat_dict=input_feat)
    # append prefix to make names global
    awq_results["daq"] += append_str_prefix(scales_list, get_op_name(model, layer) + ".")
    # Clear GPU memory
    torch.cuda.empty_cache()
    layer = layer.cpu()
    # Haotian: check activation replacement
    del input_feat
    gc.collect()
    torch.cuda.empty_cache()


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


    def _auto_get_scale(prev_op, layers, inp, module2inspect=None, kwargs={}):
        # module2inspect: if given, we will check the output diff of this module instead of layers
        if module2inspect is None:
            assert len(layers) == 1
            module2inspect = layers[0]

        scales, zp = search_module_scale(module2inspect, layers, inp, kwargs, hyper_parameters)
        # scales = scales.detach().cpu()
        # zp = zp.detach().cpu()
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


def get_loss_by_s_and_zp(module, zp, s, data_type, org_out, input_data, group=-1, **kwargs):
    org_w = module.weight.data
    old_shape = org_w.shape
    now_w = _daq_qdq_4bit(org_w, s, zp, data_type, group).reshape(*old_shape)
    # module.weight.data = now_w
    out = cal_matrix(input_data, now_w, group)
    loss = (out - org_out).float().pow(2)
    while len(loss.shape) > 1:
        loss = loss.mean(dim=0)
    # module.weight.data = org_w
    return loss


def _daq_qdq_4bit(weight, scale, zp, data_type, group=-1):
    assert data_type in FLOAT_MAPPING, "unexpected data type."
    if group > 0:
        weight = weight.view(-1, group)
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


def daq_apply_scale(module, scales_list, data_type, input_feat_dict=None):
    for prev_op_name, layer_names, scale, zp in scales_list:
        prev_op = get_op_by_name(module, prev_op_name)
        layers = [get_op_by_name(module, name) for name in layer_names]
        for fc, z, s in zip(layers, zp, scale):
            w_shape = fc.weight.shape
            if w_shape[0] != z.shape[0]:
                group = w_shape[0] * w_shape[1] // z.shape[0]
            else:
                group = -1
            fc = fc.cuda()
            fc.weight.data = _daq_qdq_4bit(fc.weight.data, s.to(fc.weight.data.device), z.to(fc.weight.data.device),
                                           data_type, group).reshape(-1, w_shape[1])
            fc = fc.cpu()


def cal_matrix(act, weight, group):
    #act [i, j, k, group] weight [-1, group] # old_version torch.matmul(x, fc_w.t())
    if group > 0:
        act = act.reshape(*act.shape[:-1],-1, group)
        weight = weight.reshape(weight.shape[0], weight.shape[1] // group , group)
        act_shp = act.shape
        assert 1 < len(act_shp) < 5
        res = torch.einsum(f'...jk,mjk->...jm', act, weight)
        res = res.transpose(-2, -1)
        return res.reshape(*res.shape[:-2],-1)
    else:
        return torch.matmul(act, weight.t())