# Enhancing consistency and mitigating bias: A data replay approach for incremental learning

The official PyTorch implementation of CwD introduced in the following paper:

> [Chenyang Wang](https://chenyang4.github.io/),  Junjun Jiang, Xingyu Hu, Xianming Liu, Xiangyang Ji;
>
> Enhancing consistency and mitigating bias: A data replay approach for incremental learning;
>
> Neural Networks, 2024.

The overall framework of the proposed method is as follows.
<div align=left>
    <img src=".\figs\framework.bmp" alt="framework" width=100%;" /> </div>

## Introduction

Deep learning systems are prone to catastrophic forgetting when learning from a sequence of tasks, as old data from previous tasks is unavailable when learning a new task. To address this, some methods propose replaying data from previous tasks during new task learning, typically using extra memory to store replay data. However, it is not expected in practice due to memory constraints and data privacy issues. Instead, data-free replay methods invert samples from the classification model. While effective, these methods face inconsistencies between inverted and real training data, overlooked in recent works. To that effect, we propose to measure the data consistency quantitatively by some simplification and assumptions. Using this measurement, we gain insight to develop a novel loss function that reduces inconsistency. Specifically, the loss minimizes the KL divergence between distributions of inverted and real data under a tied multivariate Gaussian assumption, which is simple to implement in continual learning. Additionally, we observe that old class weight norms decrease continually as learning progresses. We analyze the reasons and propose a regularization term to balance class weights, making old class samples more distinguishable. To conclude, we introduce Consistency-enhanced data replay with a Debiased classifier for class incremental learning (CwD). Extensive experiments on CIFAR-100, Tiny-ImageNet, and ImageNet100 show consistently improved performance of CwD compared to previous approaches.

## Requirements

```tex
python == 3.8
sklearn == 0.0
torch >= 1.7.0
torchvision >= 0.8
pytorch-lightning == 1.4.9
```


## Dataset
1. create a dataset root directory, e.g., data

2. cifar100 will be automatically downloaded

3. download and unzip [tiny-imagenet200](http://cs231n.stanford.edu/tiny-imagenet-200.zip) to dataset root directory

4. download and process ImageNet dataset, add 100-subset index file from [PODNet](https://github.com/arthurdouillard/incremental_learning.pytorch/tree/master/imagenet_split) to the directory

5. the overview of dataset root directory

    ```shell
    ├── cifar100
    │   └── cifar-100-python
    ├── imagenet100
    │   ├── train
    │   ├── train_100.txt
    │   ├── val
    │   └── val_100.txt
    └── tiny-imagenet200
        ├── test
        ├── train
        ├── val
        ├── wnids.txt
        └── words.txt
    ```

# Experiments


```shell
python main.py --config config/cifar100.yaml
python main.py --config config/tiny-imagenet200.yaml
python main.py --config config/imagenet100.yaml
```

Set `num_tasks` to {5, 10, 20} and change `class_order` in the config file to reproduce all the results.

## Citation

If you find our code or paper useful for your research, please cite our [paper](https://arxiv.org/abs/2401.06548).

```
@article{wang2025enhancing,
  title={Enhancing consistency and mitigating bias: A data replay approach for incremental learning},
  author={Wang, Chenyang and Jiang, Junjun and Hu, Xingyu and Liu, Xianming and Ji, Xiangyang},
  journal={Neural Networks},
  volume={184},
  pages={107053},
  year={2025},
  publisher={Elsevier}
}
```

## References 

- R-DFCIL: [Paper](https://link.springer.com/chapter/10.1007/978-3-031-20050-2_25) [Code](https://github.com/gqk/R-DFCIL)

