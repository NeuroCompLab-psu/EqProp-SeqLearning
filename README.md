# Sequence Learning using Equilibrium Propagation

The code in this repository implements the methodology described in the paper titled "Sequence Learning using Equilibrium Propagation". The fundamentals of the algorithm is described in the file model_utils.py, including computing the state and weight updates, applying the modern hopfield network in the linear projection layers, etc. The dynamics of the network as discussed in the paper is controlled by a scalar primitive function, which is defined separately for each dataset (as part of the models).  


## Environment Setup

Steps to setup conda environment:
```
1. Run the following commands
conda create --name SeqLearningEP python=3.6
conda activate SeqLearningEP
conda install -c conda-forge matplotlib
conda install pytorch torchvision -c pytorch

2. Install the following packages: tensorflow (for Keras IMDB dataset), nltk, pandas, seaborn, gensim, bs4
```

## Dataset Preprocessing
As described in the paper we have use 300D word2vec embeddings (word2vec-google-news-300) for both the datasets. For IMDB dataset the sequence length is 600, we do padding or trimming accordingly. For SNLI dataset for premise and hypothesis the maximum sequence length is restricted to 25. Modification to the sequence length is possible but corresponding change to the model parameters in model_utils.py file should be done.

##Execution

There are two primary modes of execution viz. 'train' and 'test'. We use 'train', when we want to train our model from the beginning or we want to resume training from a previously saved model `(--load-path 'saved_model/name_of_folder')`.
Few pretrained models are stored in the saved_models folder. The folder inside the saved_models folder are named as follows test_DATASETNAME_SEQLEN_ENCODDIM where sequence length is the maximum sequence length allowed in that dataset and ENCODDIM is the size of the word embeddings (usually 300). If we simply want to evaluate a previously stored model we can use `--execute test` in our list of options and choose a saved model with the `--load-path` directive.

### Implementation Specifics
As described in the paper we have used MSE loss for computing the loss function. Weight updates are done using Adam optimizer with learning decay applied (Cosine Annealing). Also we have primarily used modified three-phased EP so another option `--thirdphase` is applied which enables three-phased EP. 

### Training a convergent RNN integrated with modern hopfield network on SNLI dataset (using three-phased EP)

```
python main.py --dataset SNLI --fc_layers 400 3 --seqLen 25 --lrs 0.0005 0.0005 0.0005 0.0005 0.0002 0.0002 --lr-decay --epochs 50 --act my_hard_sig --execute train --T 60 --K 30 --batch_size 200 --alg EP --thirdphase --betas 0.0 0.5 --save --device 0 
```
The above options creates a convergent RNN with the first four layers (in parallel as described in the paper) as projection layers (lrs=0.0005, 0.0005, 0.0005, 0.0005) and the next two layers as fully connected layers of sizes as given (learning rates of each fully connected layers are listed following the projection layers). As described in the paper the modern hopfield layers is integrated with the projection layers.

### Training a convergent RNN integrated with modern hopfield network on IMDB dataset (using three-phased EP)

```
python main.py --dataset IMDB --fc_layers 1000 40 2 --seqLen 600 --lrs 0.0001 0.00005 0.00005 0.00005 --lr-decay --epochs 50 --act my_hard_sig --execute train --T 50 --K 25 --batch_size 100 --alg EP --thirdphase --betas 0.0 0.1 --save --device 0 
```

The above options creates a convergent RNN with the first layer as projection layer (lr=0.0001) and the next three as fully connected layers of sizes as given (learning rates of each fully connected layers are listed following the first projection layer). As described in the paper the modern hopfield layer is integrated with first projection layer.


## Testing a Stored Model

```
python main.py --dataset SNLI --seqLen 25 --execute test --T 60 --thirdphase --batch_size 200 --load-path saved_models/test_SNLI_25_300 --device 0
```
Above is a sample command to test a stored model. Model for IMDB was larger than the size limit allowed for submission.


## Comparing EP and BPTT Gradient estimates

We can verify the GDU theorem as referred in the paper to compare the EP and BPTT gradient estimates. We can add the option `--execute gducheck` and activate the save option `--save` inorder to create the graphs comparing the gradient estimates computed by both of them.

```
python main.py --dataset SNLI --seqLen 25 --execute gducheck --T 60 --K 30  --thirdphase --batch_size 200 --load-path saved_models/test_SNLI_25_300 --save --device 0
```
We can also run the BPTT algorithm instead of EP to train our convergent RNNs. However when the sequence size increases or T/K are comparatively larger the GPU RAM required for it to run is very high (16 GB).

## Table of the command lines arguments

|Arguments|Description|Examples|
|-------|------|------|
|`dataset`|Choose the dataset.|`--dataset 'IMDB'`, `--dataset 'SNLI'`|
|`fc_layers`|List of dimensions of fully connected layers|`--fc_layer 1000 40 2`|
|`seqLen`|Maximum Sequence length|`--seqLen 500`, `--seqLen 25`|
|`act`|Activation function for neurons|`--act 'tanh'`,`'mysig'`,`'hard_sigmoid'`|
|`execute`|Train/Test or check the theorem (gdu)|`--execute 'train'`,`--execute 'test'`, `--execute 'gducheck'`|
|`alg`|EqProp or BPTT.|`--alg 'EP'`, `--alg 'BPTT'`|
|`T`|Number of time steps in free phase.|`--T 60`|
|`K`|Number of time steps in nudge phase.|`--K 30`|
|`betas`|Beta values for free (beta1) and nudge(beta2) phases|`--betas 0.0 0.1`|
|`thirdphase`|Two nudge phases are done with beta2 and -beta2.|`--thirdphase`|
|`lrs`|Layer wise learning rates.|`--lrs 0.01 0.005`|
|`lr-decay`|Learning rate decay (Cosine Annealing).|`--lr-decay`|
|`epochs`|Number of epochs.|`--epochs 200`|
|`batch_size`|Batch size|`--batch_size 128`|
|`seed`|Random seed select.|`--seed 0`|
|`load-path`|Load a saved model.|`--load-path 'saved_models/test_SNLI_25_300'`|
|`device`|GPU Index|`--device 0`|
|`save`|Save best models (and/or plots)|`--save`|

Additional data supporting our methodology is inside the folder ./saved_model/test_SNLI_25_300 (For SNLI dataset).
