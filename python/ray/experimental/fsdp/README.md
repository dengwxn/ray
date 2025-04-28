# README

Updates 0428: It is easier to run a set of experiments using python scripts.
To customize the experiment configurations, we can specify the benchmark function like `benchmark_v1`.
To run only a specific experiment, a shell script works.

## Python Scripts

```bash
cd python/ray/experimental/fsdp
python scripts/exp_mi.py
```

## Shell Scripts

```bash
scripts/barbell_n2/llama3/torch/cc_on/fp_on/pf_on/exp_self.sh
```
