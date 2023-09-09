# train-with-fsdp

Code used to fine-tune this model: [abacaj/starcoderbase-1b-sft](https://huggingface.co/abacaj/starcoderbase-1b-sft).

Note the data in folder `data/` is not the full training data used. You can find the full set here: [evol-codealpaca-v1](https://huggingface.co/datasets/theblackcat102/evol-codealpaca-v1)

# How to run

Install dependencies:
```
python -m venv env \
  && source env/bin/activate \
  && pip install -r requirements.txt
```

Run training code:
```
torchrun --nnodes=1 --nproc-per-node=<REPLACE_WITH_NUMBER_OF_GPUS> train.py
```

To add data place jsonl files in data/ and edit `train.py` line `:154`, `:155`.

# Benchmarks
![image](https://github.com/abacaj/train-with-fsdp/assets/7272343/9c299936-c261-4992-b6d1-d61b0d6da15e)

# Charts
![image](https://github.com/abacaj/train-with-fsdp/assets/7272343/eab7e07a-f8ca-4ee3-8b33-b6e7a4016d18)

See: [wandb](https://api.wandb.ai/links/abacaj1/c4nkcs9r)
