## Coordinate Descent Attention (wip)

Implementation of an Attention layer where each head can attend to more than just one token, using coordinate descent to pick topk.

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
