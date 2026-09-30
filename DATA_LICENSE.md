# Data license

This dataset bundles MILP instances from two upstream sources, plus graph representations and labels
we computed from them. Upstream instances keep their original licenses. Our own additions (e.g. solver labels) are MIT.

| directory | source | license |
| --- | --- | --- |
| `milp_evolve/` | [MILP-Evolve](https://huggingface.co/datasets/microsoft/MILP-Evolve) ([paper](https://arxiv.org/abs/2410.08288)) | [CDLA-Permissive-2.0](https://cdla.dev/permissive-2-0/) |
| `dmiplib/{CA,IS,SC,VC,GISP,MMCN,CFLP,OTS,NNV}/` | [Distributional MIPLIB](https://huggingface.co/datasets/weiminhu/D-MIPLIB) ([website](https://sites.google.com/usc.edu/distributional-miplib), [paper](https://arxiv.org/abs/2406.06954)) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode) |
| `dmiplib/LB/` | D-MIPLIB, from the [ML4CO competition](https://github.com/ds4dm/ml4co-competition) | [BSD 3-Clause](https://github.com/ds4dm/ml4co-competition/blob/main/LICENSE) |

Notes:

- D-MIPLIB has no single license: each domain is licensed according to where it was curated from. See the [D-MIPLIB website](https://sites.google.com/usc.edu/distributional-miplib) for the original source of each domain.
- The graphs (`graphs/*.pt.zst`) are converted from the upstream instances, and `raw.tar` contains
the instances themselves. Both are modified material and carry the upstream terms.
- Where terms differ, the upstream license governs the affected files.
