## Coordinate Descent Attention (wip)

Implementation of an Attention layer where each head can attend to more than just one token, using coordinate descent to pick topk. Perhaps the number of tokens to attend to can even be learned.

<a href="https://api.wandb.ai/links/lucidrains/7amjt5kw">Ongoing experiments</a>

## Install

```bash
$ pip install coordinate-descent-attention
```

## Usage

```python
import torch
from coordinate_descent_attention import Transformer

model = Transformer(
    num_tokens = 256,
    dim = 512,
    depth = 2,
    seq_len = 2048,
    dim_head = 64,
    heads = 8,
    use_coor_descent = True   # set to True to switch from softmax to coordinate descent on qk similarity matrix
).cuda()

x = torch.randint(0, 256, (1, 2048)).cuda()

logits = model(x)
```

## Citations

```bibtex
@article{Wright2015CoordinateDA,
    title   = {Coordinate descent algorithms},
    author  = {Stephen J. Wright},
    journal = {Mathematical Programming},
    year    = {2015},
    volume  = {151},
    pages   = {3-34}
}
```

```bibtex
@inproceedings{Gupta2021MemoryefficientTV,
    title   = {Memory-efficient Transformers via Top-k Attention},
    author  = {Ankit Gupta and Guy Dar and Shaya Goodman and David Ciprut and Jonathan Berant},
    booktitle = {SUSTAINLP},
    year    = {2021}
}
```

```bibtex
@article{Zhao2019ExplicitST,
    title   = {Explicit Sparse Transformer: Concentrated Attention Through Explicit Selection},
    author  = {Guangxiang Zhao and Junyang Lin and Zhiyuan Zhang and Xuancheng Ren and Qi Su and Xu Sun},
    journal = {ArXiv},
    year    = {2019},
    volume  = {abs/1912.11637}
}
```
