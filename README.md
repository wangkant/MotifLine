# motifline

ChromBPNet attributions -> TF-MoDISco -> FiNeMo, in Snakemake.

```bash
conda env create -p envs/compute -f envs/compute.yaml
conda env create -p envs/orchestration -f envs/orchestration.yaml
$EDITOR config/config.yaml
envs/orchestration/bin/snakemake -j 8 --resources gpu=1
```
