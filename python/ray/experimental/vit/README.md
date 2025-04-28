# README

Updates 0428: Add README scripts.
To vary the configurations for Ray, update the parameters within the code, like `num_dp_vision`, `num_dp_text`, `accelerator_type="H100"/"TITAN"`.
Before experiments, we need to start a Ray master node on one machine and let the other machine connect to this.
Note that two nodes should run in the same environments with the same dependencies, otherwise Ray might not work.
The easiest way is to use the container image under `container/`.

## Scripts

```bash
# Ray
scripts/v1/lld.sh

# PyTorch
scripts/v3/lld.sh
```
