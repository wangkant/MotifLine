#!/bin/bash
# Environments in .envs/, the AlphaGenome model code into the compute env, reference data
# and AlphaGenome weights in data/. Rerunning skips what is already there.
set -euo pipefail
cd "$(dirname "$0")"
CONDA=$(command -v mamba || command -v conda)

for e in orchestration compute meme chrombpnet; do
    [ -x .envs/$e/bin/python ] || $CONDA env create -q -y -p "$PWD/.envs/$e" -f envs/$e.yaml
done
# the AlphaGenome model code is pinned here; its dependencies are pinned in envs/compute.yaml
.envs/compute/bin/pip install -q --no-deps --ignore-requires-python \
    "alphagenome-pytorch @ git+https://github.com/rastogiruchir/alphagenome-pytorch.git@64e9b75d3d1e34a4e212b94390ed8bde6fe07d9e"

mkdir -p data
get() {
    if [ ! -s "data/$1" ]; then
        curl -fL -# --retry 3 -o "data/$1.part" "$2"
        mv "data/$1.part" "data/$1"
    fi
}
# hg38: ENCODE's GRCh38 no-alt analysis set
ENC=https://www.encodeproject.org/files
if [ ! -s data/hg38.fa ]; then
    curl -fL -# --retry 3 $ENC/GRCh38_no_alt_analysis_set_GCA_000001405.15/@@download/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta.gz \
        | gunzip > data/hg38.fa.part
    mv data/hg38.fa.part data/hg38.fa
fi
get hg38.chrom.sizes $ENC/GRCh38_EBV.chrom.sizes/@@download/GRCh38_EBV.chrom.sizes.tsv
H14=https://hocomoco14.autosome.org/final_bundle/hocomoco14/H14CORE
get H14CORE_meme_format.meme $H14/formatted_motifs/H14CORE_meme_format.meme
get H14CORE_annotation.jsonl $H14/H14CORE_annotation.jsonl
get JASPAR2026_CORE_vertebrates_non-redundant_pfms_meme.txt \
    https://jaspar.elixir.no/download/data/2026/CORE/JASPAR2026_CORE_vertebrates_non-redundant_pfms_meme.txt
get alphagenome_model_fold_0.safetensors \
    https://huggingface.co/gtca/alphagenome_pytorch/resolve/main/model_fold_0.safetensors
