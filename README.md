# DAQ: Density-Aware Post-Training Weight-Only Quantization For LLMs

## Install

1. Clone this repository and navigate to DAQ folder
```bash
git clone http://this/repo
cd daq
```

2. Install Package
```bash
conda create -n daq python=3.10 -y
conda activate daq
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

## Usage
To run DAQ:
```bash
python awq/entry.py --sample 1 --model_path /Llama-2-7b-hf --run_daq --tasks wikitext --w_bit 4 --q_group_size -1 --q_backend fake --dump daq_cache/Llama-2-7b-hf-daq-sym-dca-channel-nf4.pt
```

To run DAQ+AWQ:

```bash
python awq/entry.py --model_path /Llama-2-7b-hf --calibration daq --run_awq --tasks wikitext --w_bit 4 --q_group_size -1 --q_backend fake --dump awq_cache/Llama-2-7b-hf-daq-sym-dca-channel-nf4.pt --sample 2 --data_type nf4
```

## Acknowledgements
We would like to express our gratitude to the AWQ project for their pioneering work in weight quantization for LLMs. Our work builds upon their insights and implementations.