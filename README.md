# Tri-CDR
The source code is for the paper: [Triple Sequence Learning for Cross-domain Recommendation](https://arxiv.org/pdf/2304.05027.pdf) accepted in TOIS by Haokai Ma, Ruobing Xie, Lei Meng, Xin Chen, Xu Zhang, Leyu Lin and Jie Zhou.

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
Due to an imminent submission deadline, the related codes will be organized and made available after this submission process.

## BibTeX
If you find this work useful for your research, please kindly cite Tri-CDR by:
```
@misc{ma2023triple,
      title={Triple Sequence Learning for Cross-domain Recommendation}, 
      author={Haokai Ma and Ruobing Xie and Lei Meng and Xin Chen and Xu Zhang and Leyu Lin and Jie Zhou},
      year={2023},
      eprint={2304.05027},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
```

