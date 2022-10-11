# Demo code of TOAK

This is the demo code for the paper "Adversarial for Social Privacy: A Poisoning Strategy to Degrade User Identity Linkage". It contains the following parts:

* toak_attack.py : source code of proposed TOAK.
* vgae.py : source code of Variational Graph Auto-encoder (VGAE), which is used to generate node embedding.
* dataset/ : a folder for storing dataset. currently, we provide the Douban dataset and will upload the TF and ARXIV datasets later.

The commond for run the TOAK is:

```
python toak_attack.py --dataset=douban  
```


after running, the flipped edge set will be stored at ./attack_graph/douban/toak/
