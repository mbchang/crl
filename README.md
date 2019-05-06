# Compositional Recursive Learner

This repo contains the code for the paper [Automatically Composing Representation Transformations as a Means for Generalization](https://arxiv.org/abs/1807.04640).

```
@article{chang2018automatically,
  title={Automatically composing representation transformations as a means for generalization},
  author={Chang, Michael B and Gupta, Abhishek and Levine, Sergey and Griffiths, Thomas L},
  journal={arXiv preprint arXiv:1807.04640},
  year={2018}
}
```

![](https://github.com/mbchang/crl/blob/master/figs/main_diagram_final.png)

## Running the Code

For multilingual arithmetic, run the following command: `python crl_arithlang.py`

For MNIST image transformations, run the following command: `python crl_imagetransform.py`

You can print to a text file by adding the argument `--printf`.

If you are just want a quick run-through to see how the code works, add the argument `--debug`.
