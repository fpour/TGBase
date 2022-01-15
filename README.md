# A Strong Node Classification Baseline for Temporal Graphs


## Introduction
Representation learning on temporal networks is a first step for performing further analysis, e.g. node classification.
We propose **TGBase** that extracts key features to consider the structural characteristics of each node and its neighborhood as well as the intensity and timestamp of the interactions among node pairs.


#### Paper link: [A Strong Node Classification Baseline for Temporal Graphs](link will be added.)




## Experiments

### Requirements
```{bash}
pandas==1.1.0
torch==1.10.0
scikit_learn==1.0.1
numpy==1.21.2
```
### Datasets
#### Public data
Sample datasets can be downloaded from the following sources:
* Networks with static labels:
  * Cryptocurrency transaction networks: [Bitcoin](https://www.kaggle.com/ellipticco/elliptic-data-set), [Ethereum](https://www.kaggle.com/xblock/ethereum-phishing-transaction-network)
  * [Rating platforms](https://cs.stanford.edu/~srijan/rev2/) (e.g., Amazon, OTC)
* Networks with dynamic labels:
  * [Social networks](http://snap.stanford.edu/jodie/) (e.g., Wikipedia, Reddit)

To user your own data, it should have similar format to the above datasets.
All data are assumed to be in "_data_" folder.

### Execution
* Static node classification: 
  * To generate TGBase embedding for _OTC_ dataset and classify the nodes with a Random Forest classifier:
    ```{bash}
    python src/TGBase_staticEmb.py --network otc
    python src/static_n_clf.py --network otc --clf RF
    ```
* Dynamic node classification:
  * To generate embeddings for wikipedia network and apply the classification with a MLP classifier:
    ```{bash}
    python src/TGBase_DynEmb.py --network wikipedia
    python src/dynamic_n_clf.py --network wikipedia --clf MLP
    ```
An execution summary is saved in "_logs_" folder.

### Acknowledgement
We would like to thank the authors of [TGN](https://github.com/twitter-research/tgn) for providing open access to the implementation of their methods.
 
## Cite us

```bibtex
@inproceedings{tgbase_sdm_2022,
    title={A Strong Node Classification Baseline for Temporal Graphs},
    author={Farimah Poursafaei and Zeljko Zilic and Reihaneh Rabbany},
    booktitle={SIAM International Conference on Data Mining (SIAM SDM22)},
    year={2022}
}
```





