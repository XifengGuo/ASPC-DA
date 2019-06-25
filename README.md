# Adaptive Self-paced Deep Clustering with Data Augmentation (ASPC-DA)

Tensorflow implementation for our paper:
* Xifeng Guo, Xinwang Liu, En Zhu, et al. [Adaptive Self-paced Deep Clustering with Data Augmentation](https://xifengguo.github.io/papers/TKDE19-ASPC-DA.pdf). 
IEEE Transactions on Knowledge and Data Engineering (TKDE), 2019.

## Abstract
Deep clustering gains superior performance than conventional clustering by jointly performing feature learning and cluster
assignment. Although numerous deep clustering algorithms have emerged in various applications, most of them fail to learn robust
cluster-oriented features which in turn hurts the final clustering performance. To solve this problem, we propose a two-stage deep
clustering algorithm by incorporating data augmentation and self-paced learning. Specifically, in the first stage, we learn robust features
by training an autoencoder with examples that are augmented by random shifting and rotating the given clean examples. Then in the
second stage, we encourage the learned features to be cluster-oriented by alternatively finetuning the encoder with the augmented
examples and updating the cluster assignments of the clean examples. During finetuning the encoder, the target of each augmented
example in the loss function is the center of the cluster to which the clean example is assigned. The targets may be computed
incorrectly, and the examples with incorrect targets could mislead the encoder network. To stabilize the network training, we select
most confident examples in each iteration by utilizing the adaptive self-paced learning. Extensive experiments validate that our
algorithm outperforms the state of the arts on four image datasets.


## Usage

### 1. Prepare environment

Install [Anaconda](https://www.anaconda.com/download/) with Python 3.6 version (_Optional_).   
Create a new env (_Optional_):   
```
conda create -n aspc python=3.6 -y   
source activate aspc  # Linux 
#  or 
conda activate aspc  # Windows
```
Install required packages:
```
pip install tensorflow-gpu==1.10 scikit-learn h5py  
```
### 2. Clone the code and prepare the datasets.

```
git clone https://github.com/XifengGuo/ASPC-DA.git ASPC-DA
cd ASPC-DA
```

### 3. Run experiments.    

```bash
python run_exp.py
```