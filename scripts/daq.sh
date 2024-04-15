export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

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

python awq/entry.py --model_path /root/autodl-tmp/opt --run_daq --tasks wikitext --w_bit 4 --q_group_size -1 --q_backend fake --dump_awq awq_cache/opt-6.7b-w4-channel-sum.pt
python awq/entry.py --model_path /root/autodl-tmp/opt --run_awq --tasks wikitext,piqa --w_bit 4 --q_group_size -1 --q_backend fake --dump awq_cache/awq/opt-6.7b-w4-channel.pt

python awq/entry.py --model_path /root/autodl-tmp/llama-7b/models--meta-llama--Llama-2-7b-hf/snapshots/8cca527612d856d7d32bd94f8103728d614eb852 --run_daq --tasks wikitext --w_bit 4 --q_group_size -1 --q_backend fake --dump_awq awq_cache/llama2-7b-w4-channel-no-sum.pt

python awq/entry.py --model_path /root/autodl-tmp/llama-7b/models--facebook--opt-13b/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5 --run_daq --tasks wikitext --w_bit 4 --q_group_size -1 --q_backend fake --dump awq_cache/opt-13b-w4-channel-no-sum.pt


python awq/entry.py --model_path /root/autodl-tmp/llama-7b/models--facebook--opt-2.7b/snapshots/905a4b602cda5c501f1b3a2650a4152680238254 --tasks wikitext,lambada_openai,piqa,hellaswag --w_bit 4 --q_group_size -1 --q_backend fake --load awq_cache/opt-2.7b-w4-channel-sum.pt
python awq/entry.py --model_path /root/autodl-tmp/opt --run_awq --tasks wikitext,lambada_openai --w_bit 4 --q_group_size -1 --q_backend fake --dump awq_cache/awq/opt-6.7b-w4-channel.pt
python awq/entry.py --model_path /root/autodl-tmp/llama-7b/models--facebook--opt-13b/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5 --run_awq --tasks wikitext,lambada_openai --w_bit 4 --q_group_size -1 --q_backend fake --dump awq_cache/awq/opt-13b-w4-channel.pt
python awq/entry.py --model_path /root/autodl-tmp/llama-7b/models--meta-llama--Llama-2-13b-hf/snapshots/dc1d3b3bfdb69df26f8fc966c16353274b138c55 --run_awq --tasks wikitext,lambada_openai --w_bit 4 --q_group_size -1 --q_backend fake --dump awq_cache/awq/llama-13b-w4-channel.pt
python awq/entry.py --model_path /root/autodl-tmp/llama-7b/models--meta-llama--Llama-2-7b-hf/snapshots/8cca527612d856d7d32bd94f8103728d614eb852 --run_awq --tasks wikitext,lambada_openai --w_bit 4 --q_group_size -1 --q_backend fake --dump awq_cache/awq/llama-2-7b-w4-channel.pt

python awq/entry.py --model_path /root/autodl-tmp/opt --run_daq --tasks wikitext --w_bit 4 --q_group_size -1 --q_backend fake --dump daq_cache/llama2-7b-w4-channel-no-sum-no-dca.pt
python awq/entry.py --model_path /root/autodl-tmp/llama-7b/models--meta-llama--Llama-2-7b-hf/snapshots/8cca527612d856d7d32bd94f8103728d614eb852 --run_daq --tasks wikitext --w_bit 4 --q_group_size -1 --q_backend fake --dump daq_cache/opt-6.7b-w4-channel-no-sum-no-dca.pt

python awq/entry.py --model_path /root/autodl-tmp/abc/8a0442e81540efaeb1a0fe3e95477b5e0edfd423 --run_daq --tasks wikitext --w_bit 4 --q_group_size 256 --q_backend fake --dump daq_cache/llama-2-7b-w4-256-no-sum-no-dca.pt


/public/home/acfanoll2u/miniconda3/envs/py39/bin/python3 awq/entry.py --model_path /public/home/acfanoll2u/dataset/llama7b/Llama-2-7b-hf --run_daq --tasks wikitext --w_bit 4 --q_group_size 256 --q_backend fake --dump daq_cache/Llama-2-7b-w4-256-v2.pt

