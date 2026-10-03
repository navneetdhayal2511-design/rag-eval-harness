# Ablation

| pipeline | recall@budget | ndcg@budget | recall@5 | ndcg@5 | mrr | citation_precision | judge_correct | chunks_in_budget | latency_ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bm25-fixed | 0.767 | 0.831 | 0.942 | 0.879 | 0.885 | 0.783 | 0.833 | 1.3 | 3.0 |
| bm25-section | 0.933 | 0.870 | 0.950 | 0.876 | 0.858 | 0.400 | 0.700 | 4.5 | 4.1 |
| dense-section | 0.933 | 0.789 | 0.933 | 0.789 | 0.749 | 0.386 | 0.717 | 4.9 | 3.9 |
| hybrid-weighted | 0.933 | 0.868 | 0.933 | 0.868 | 0.856 | 0.386 | 0.700 | 4.8 | 8.8 |
| hybrid-rrf | 0.933 | 0.860 | 0.933 | 0.860 | 0.845 | 0.392 | 0.700 | 4.7 | 7.8 |
| hybrid-rrf-mmr | 0.925 | 0.779 | 0.925 | 0.779 | 0.744 | 0.394 | 0.733 | 5.0 | 44.3 |


# `bm25-section` vs `bm25-fixed`

Paired over the questions both runs cover. `p` is a two-sided sign-flip
permutation test; a delta whose interval spans zero is noise.

| metric | baseline | candidate | delta | 95% CI | p | verdict |
| --- | ---: | ---: | ---: | :---: | ---: | --- |
| recall@budget | 0.767 | 0.933 | +0.167 | [+0.092, +0.250] | 0.000 | improved |
| ndcg@budget | 0.831 | 0.870 | +0.039 | [-0.022, +0.101] | 0.241 | no change |
| recall@5 | 0.942 | 0.950 | +0.008 | [-0.059, +0.075] | 1.000 | no change |
| ndcg@5 | 0.879 | 0.876 | -0.003 | [-0.052, +0.042] | 0.890 | no change |
| mrr | 0.885 | 0.858 | -0.026 | [-0.077, +0.021] | 0.312 | no change |
| citation_precision | 0.783 | 0.400 | -0.383 | [-0.461, -0.306] | 0.000 | regressed |
| judge_correct | 0.833 | 0.700 | -0.133 | [-0.233, -0.033] | 0.022 | regressed |
| chunks_in_budget | 1.3 | 4.5 | +3.133 | [+2.883, +3.400] | 0.000 | improved |
| latency_ms | 3.0 | 4.1 | +1.097 | [-1.203, +3.226] | 0.351 | no change |


# `dense-section` vs `bm25-fixed`

Paired over the questions both runs cover. `p` is a two-sided sign-flip
permutation test; a delta whose interval spans zero is noise.

| metric | baseline | candidate | delta | 95% CI | p | verdict |
| --- | ---: | ---: | ---: | :---: | ---: | --- |
| recall@budget | 0.767 | 0.933 | +0.167 | [+0.092, +0.250] | 0.000 | improved |
| ndcg@budget | 0.831 | 0.789 | -0.042 | [-0.113, +0.033] | 0.285 | no change |
| recall@5 | 0.942 | 0.933 | -0.008 | [-0.067, +0.042] | 1.000 | no change |
| ndcg@5 | 0.879 | 0.789 | -0.090 | [-0.152, -0.028] | 0.005 | regressed |
| mrr | 0.885 | 0.749 | -0.136 | [-0.206, -0.065] | 0.000 | regressed |
| citation_precision | 0.783 | 0.386 | -0.397 | [-0.472, -0.319] | 0.000 | regressed |
| judge_correct | 0.833 | 0.717 | -0.117 | [-0.217, -0.033] | 0.038 | regressed |
| chunks_in_budget | 1.3 | 4.9 | +3.600 | [+3.367, +3.833] | 0.000 | improved |
| latency_ms | 3.0 | 3.9 | +0.872 | [-0.969, +2.772] | 0.364 | no change |


# `hybrid-weighted` vs `bm25-fixed`

Paired over the questions both runs cover. `p` is a two-sided sign-flip
permutation test; a delta whose interval spans zero is noise.

| metric | baseline | candidate | delta | 95% CI | p | verdict |
| --- | ---: | ---: | ---: | :---: | ---: | --- |
| recall@budget | 0.767 | 0.933 | +0.167 | [+0.092, +0.250] | 0.000 | improved |
| ndcg@budget | 0.831 | 0.868 | +0.037 | [-0.024, +0.102] | 0.262 | no change |
| recall@5 | 0.942 | 0.933 | -0.008 | [-0.067, +0.042] | 1.000 | no change |
| ndcg@5 | 0.879 | 0.868 | -0.011 | [-0.060, +0.034] | 0.651 | no change |
| mrr | 0.885 | 0.856 | -0.029 | [-0.079, +0.020] | 0.268 | no change |
| citation_precision | 0.783 | 0.386 | -0.397 | [-0.472, -0.322] | 0.000 | regressed |
| judge_correct | 0.833 | 0.700 | -0.133 | [-0.233, -0.033] | 0.022 | regressed |
| chunks_in_budget | 1.3 | 4.8 | +3.450 | [+3.217, +3.683] | 0.000 | improved |
| latency_ms | 3.0 | 8.8 | +5.727 | [+2.933, +8.704] | 0.000 | regressed |


# `hybrid-rrf` vs `bm25-fixed`

Paired over the questions both runs cover. `p` is a two-sided sign-flip
permutation test; a delta whose interval spans zero is noise.

| metric | baseline | candidate | delta | 95% CI | p | verdict |
| --- | ---: | ---: | ---: | :---: | ---: | --- |
| recall@budget | 0.767 | 0.933 | +0.167 | [+0.092, +0.250] | 0.000 | improved |
| ndcg@budget | 0.831 | 0.860 | +0.029 | [-0.034, +0.093] | 0.396 | no change |
| recall@5 | 0.942 | 0.933 | -0.008 | [-0.067, +0.042] | 1.000 | no change |
| ndcg@5 | 0.879 | 0.860 | -0.019 | [-0.071, +0.028] | 0.449 | no change |
| mrr | 0.885 | 0.845 | -0.040 | [-0.096, +0.015] | 0.160 | no change |
| citation_precision | 0.783 | 0.392 | -0.392 | [-0.467, -0.314] | 0.000 | regressed |
| judge_correct | 0.833 | 0.700 | -0.133 | [-0.233, -0.033] | 0.022 | regressed |
| chunks_in_budget | 1.3 | 4.7 | +3.350 | [+3.117, +3.600] | 0.000 | improved |
| latency_ms | 3.0 | 7.8 | +4.806 | [+2.035, +8.128] | 0.002 | regressed |


# `hybrid-rrf-mmr` vs `bm25-fixed`

Paired over the questions both runs cover. `p` is a two-sided sign-flip
permutation test; a delta whose interval spans zero is noise.

| metric | baseline | candidate | delta | 95% CI | p | verdict |
| --- | ---: | ---: | ---: | :---: | ---: | --- |
| recall@budget | 0.767 | 0.925 | +0.158 | [+0.083, +0.242] | 0.000 | improved |
| ndcg@budget | 0.831 | 0.779 | -0.052 | [-0.131, +0.027] | 0.219 | no change |
| recall@5 | 0.942 | 0.925 | -0.017 | [-0.083, +0.033] | 0.784 | no change |
| ndcg@5 | 0.879 | 0.779 | -0.100 | [-0.165, -0.035] | 0.003 | regressed |
| mrr | 0.885 | 0.744 | -0.140 | [-0.215, -0.068] | 0.000 | regressed |
| citation_precision | 0.783 | 0.394 | -0.389 | [-0.469, -0.306] | 0.000 | regressed |
| judge_correct | 0.833 | 0.733 | -0.100 | [-0.200, +0.000] | 0.113 | no change |
| chunks_in_budget | 1.3 | 5.0 | +3.633 | [+3.400, +3.867] | 0.000 | improved |
| latency_ms | 3.0 | 44.3 | +41.307 | [+37.200, +45.775] | 0.000 | regressed |
