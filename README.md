# This is the pytorch implementation of the paper "Tackling Multiplayer Interaction for Federated Generative Adversarial Networks".
----
This project bases on [PFLlib: Personalized Federated Learning Algorithm Library](https://github.com/TsingZ0/PFLlib). We modify sending and aggregation functions to support training federated Generative Adversarial Networks (federated GANs). We call our federated GAN system Oasis for ***O***rchestrating Gener***a***tive Adversarial Network***s*** in Federated Learn***i***ng Paradigm***s***.   
    
So far, Oasis supports GAN models including [MLPGAN](https://papers.nips.cc/paper_files/paper/2014/hash/5ca3e9b122f61f8f06494c97b1afccf3-Abstract.html), [DCGAN](https://arxiv.org/abs/1511.06434) and [U-NetGAN](https://openaccess.thecvf.com/content_CVPR_2020/papers/Schonfeld_A_U-Net_Based_Discriminator_for_Generative_Adversarial_Networks_CVPR_2020_paper.pdf). 
       
Oasis implements four Federated Learning Algorithm: [FedAvg](https://arxiv.org/abs/1602.05629), [FedProx](https://arxiv.org/abs/1812.06127), [Ditto](https://arxiv.org/abs/2012.04221), and Ours (in this project, we call it FedSim and FedSim2).     

The datasets include [MNIST](https://yann.lecun.com/exdb/mnist/), [CIfar10](https://www.cs.toronto.edu/~kriz/cifar.html), and [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), which can be downloaded from their offical website.   

## Project Architecture
The folder `dataset` includes the code to split data into clients.   

The folder `system` is Oasis core code. Inside it, the folder `clients` and `servers` are the core algorithms. The folder `stl` includes the `Self-taight Learning` code. The folder `trainmodel` includes the GAN models we used.     

## How to Start
A simple example:
```
cd ./system
python main.py -data mnist -algo FedAvg -gr 2500 -ls 5
```
For more parameter settings please see the `argparse` in the `main.py`. 