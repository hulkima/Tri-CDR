# Tri-CDR
The source code is for the paper: [Triple Sequence Learning for Cross-domain Recommendation](https://dl.acm.org/doi/pdf/10.1145/3638351) accepted in TOIS by Haokai Ma, Ruobing Xie, Lei Meng, Xin Chen, Xu Zhang, Leyu Lin and Jie Zhou.

## Overview
This paper presents a novel framework, termed triple sequence learning for cross-domain recommendation (Tri-CDR), which jointly models the source, target, and mixed behavior sequences to highlight the global and target preference and precisely model the triple correlation in CDR. Specifically, Tri-CDR independently models the hidden representations for the triple behavior sequences and proposes a triple cross-domain attention (TCA) method to emphasize the informative knowledge related to both user's global and target-domain preference. To comprehensively explore the cross-domain correlations, we design a triple contrastive learning (TCL) strategy that simultaneously considers the coarse-grained similarities and fine-grained distinctions among the triple sequences, ensuring the alignment while preserving information diversity in multi-domain.![_](./structure.png)

## Dependencies
- Python 3.8.10
- PyTorch 1.12.0+cu102
- pytorch-lightning==1.6.5
- Torchvision==0.8.2
- Pandas==1.3.5
- Scipy==1.7.3

## Implementation of Tri-CDR
For the Game->Toy setting:
```
CUDA_VISIBLE_DEVICES=0 python Tri_CDR.py --cross_dataset=Toy_Game --dataset=amazon_toy --rate_mix_source 1 --rate_mix_target 1 --rate_source_target 1 --cl_weight 0.1 --triplet_weight 10.0 --triplet_margin 4.0
```

For the Toy->Game setting:
```
CUDA_VISIBLE_DEVICES=0 python Tri_CDR.py --cross_dataset=Toy_Game --dataset=amazon_game --rate_mix_source 1000 --rate_mix_target 1 --rate_source_target 1 --cl_weight 4.0 --triplet_weight 5.0 --triplet_margin 20.0
```
To achieve quick deployment, we just implemented TCA with the simplest Loop approach, which does not impact the performance but may require additional GPU space. I will update this part to batch processing when time allows. 

## BibTeX
If you find this work useful for your research, please kindly cite Tri-CDR by:
```
@article{Tri-CDR,
      author = {Ma, Haokai and Xie, Ruobing and Meng, Lei and Chen, Xin and Zhang, Xu and Lin, Leyu and Zhou, Jie},
      title = {Triple Sequence Learning for Cross-domain Recommendation},
      year = {2023},
      publisher = {Association for Computing Machinery},
      journal = {ACM Trans. Inf. Syst. (TOIS)},
}
```

