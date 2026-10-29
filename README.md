# Wired for Overconfidence: A Mechanistic Perspective on Inflated Verbalized Confidence in LLMs

## Setup

```bash
conda create -n mi-miscalib python=3.10 -y && conda activate mi-miscalib
pip install -r requirements.txt
pip install -e ".[dev]"
pytest      
```

## Quick start

```bash
bash scripts/run_pipeline.sh Qwen/Qwen2.5-3B-Instruct popqa cuda:0
```

## Acknowledgements

`src/eap` is adapted from [EAP-IG](https://github.com/hannamw/EAP-IG)
(Hanna et al., COLM 2024).
