# Active Learning on a Budget - Opposite Strategies Suit High and Low Budgets

This is the official implementation for the paper **Active Learning on a Budget - Opposite Strategies Suit High and Low Budgets**. This code implements TypiClust - a Simple and Effective Low Budget Active Learning method.

[**Arxiv link**](https://arxiv.org/abs/2202.02794), 
[**Twitter Post link**](https://twitter.com/AvihuDkl/status/1529385835694637058)


TypiClust first employs a representation learning method, then clusters the data into K clusters, and selects the most Typical (Dense) sample from every cluster. In other words, TypiClust selects samples from dense and diverse regions of the data distribution.

<img src="https://user-images.githubusercontent.com/39214195/161326609-a60ff5fc-ca97-4fdb-bd28-5a5468c2499c.png" width=440>

![cifar_selection.png](cifar_selection.png)
![TypiClust.gif](2d_selection_gif.gif)

<img src="./2d_selection_gif.gif" width="440">

The method is examined using the evaluation framework built by Munjal et al. 

![results.png](results.png)!

## Introduction

The goal of this repository is to provide a simple and flexible codebase for deep active learning. It is designed to support rapid implementation and evaluation of research ideas. We also provide a results on CIFAR10 below.

## Using the Toolkit

Please see [`GETTING_STARTED`](deep-al/docs/GETTING_STARTED.md) for brief instructions on installation, adding new datasets, basic usage examples.


## Citing this Repository

If you find this repo helpful in your research, please consider citing us and the owners of the original toolkit:

@article{Chandra2021DeepAL,
    Author = {Akshay L Chandra and Vineeth N Balasubramanian},
    Title = {Deep Active Learning Toolkit for Image Classification in PyTorch},
    Journal = {https://github.com/acl21/deep-active-learning-pytorch},
    Year = {2021}
}

@article{Munjal2020TowardsRA,
  title={Towards Robust and Reproducible Active Learning Using Neural Networks},
  author={Prateek Munjal and N. Hayat and Munawar Hayat and J. Sourati and S. Khan},
  journal={ArXiv},
  year={2020},
  volume={abs/2002.09564}
}
```

## License

This toolkit is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.
