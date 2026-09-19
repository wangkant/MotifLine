# motifline

ChromBPNet and AlphaGenome attributions -> TF-MoDISco -> FiNeMo, in Snakemake.
Patterns are named by TomTom against HOCOMOCO H14 CORE + JASPAR CORE vertebrates.

```bash
conda env create -p envs/compute -f envs/compute.yaml
conda env create -p envs/orchestration -f envs/orchestration.yaml
conda env create -p envs/meme -f envs/meme.yaml
git clone -b deeplift-support https://github.com/rastogiruchir/alphagenome-pytorch.git
$EDITOR config/config.yaml
envs/orchestration/bin/snakemake -j 8 --resources gpu=4
```

`envs/locks/` holds the full package lists of the environments the runs used.
