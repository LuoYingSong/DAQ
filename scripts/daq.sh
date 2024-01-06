--model_path /root/autodl-tmp/models--facebook--opt-125m/snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6 --run_daq --tasks wikitext --w_bit 4 --q_group_size -1 --q_backend fake --dump_daq awq_cache/opt-2.7b-w4-g128.pt
--model_path /root/autodl-tmp/models--facebook--opt-125m/snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6  --tasks wikitext --w_bit 4 --q_group_size -1 --q_backend fake --load_daq awq_cache/opt-2.7b-w4-g128.pt
--model_path /root/autodl-tmp/llama-7b/models--meta-llama--Llama-2-7b-hf/snapshots/8cca527612d856d7d32bd94f8103728d614eb852 --run_daq --tasks wikitext --w_bit 4 --q_group_size -1 --q_backend fake --dump_daq awq_cache/llama2-7b-w4-channel.pt
--model_path /home/lys/.cache/huggingface/hub/models--facebook--opt-125m/snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6 --run_daq --tasks wikitext --w_bit 4 --q_group_size -1 --q_backend fake --dump_daq awq_cache/opt-125m-w4-channel.pt
--model_path /root/autodl-tmp/opt --run_daq --tasks wikitext --w_bit 4 --q_group_size -1 --q_backend fake --dump_daq awq_cache/opt-6.7b-w4-channel.pt
--model_path /root/autodl-tmp/models--facebook--opt-125m/snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6 --run_daq --tasks wikitext --w_bit 4 --q_group_size 128 --q_backend fake --dump_daq awq_cache/opt-125m-w4-channel.pt

python awq/entry.py --model_path /root/autodl-tmp/models--facebook--opt-125m/snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6 --run_daq --tasks wikitext --w_bit 4 --q_group_size 128 --q_backend fake --dump_daq awq_cache/opt-125m-w4-128g.pt
python awq/entry.py --model_path /root/autodl-tmp/llama-7b/models--meta-llama--Llama-2-7b-hf/snapshots/8cca527612d856d7d32bd94f8103728d614eb852 --run_daq --tasks wikitext --w_bit 4 --q_group_size 128 --q_backend fake --dump_daq awq_cache/llama2-7b-w4-128g.pt


python awq/entry.py --model_path /root/autodl-tmp/opt --run_daq --tasks wikitext --w_bit 4 --q_group_size -1 --q_backend fake --dump_daq awq_cache/opt-6.7b-w4-channel.pt

# 模型加载命令
python awq/entry.py --model_path /root/autodl-tmp/opt --tasks wikitext2 --w_bit 4 --q_group_size -1 --q_backend fake --load_daq awq_cache/opt-6.7b-w4-channel.pt
# 模型优化命令
python awq/entry.py --model_path /root/autodl-tmp/opt --run_daq --tasks wikitext --w_bit 4 --q_group_size 128 --q_backend fake --dump_daq awq_cache/opt-125m-w4-128.pt

python awq/entry.py --model_path /root/autodl-tmp/opt --run_daq --tasks wikitext --w_bit 4 --q_group_size 128 --q_backend fake --dump_daq awq_cache/opt-6.7b-w4-128g.pt