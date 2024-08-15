# DAQ: Density-Aware Post-Training Weight-Only Quantization For LLMs

## Install

1. Clone this repository and navigate to DAQ folder
```bash
git clone xxx
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
To run DAQ on the model:
```bash
python awq/entry.py --sample 1 --model_path /root/autodl-tmp/opt --run_daq --tasks wikitext --w_bit 4 --q_group_size -1 --q_backend fake --dump daq_cache/opt-6.7b-w4-channel-1block.pt
```

## Acknowledgements
We would like to express our gratitude to the AWQ project for their pioneering work in weight quantization for LLMs. Our work builds upon their insights and implementations.