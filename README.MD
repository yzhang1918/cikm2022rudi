# RuDi

Codes and data for KDD2022 submission.

## Usage

Un-tar datasets (Due to the large size, we do not include the proposed Elo. One can download the raw dataset and use our provided pre-processing code.)
```
tar zxf data.tar.gz
```

Run the algorithm:
```
chmod +x run.sh
./run.sh data/vews_all cuda:0 gru
```

Then the statistics and rules are saved in `data/[dataset]/[teacher]_stats` and `data/[dataset]/[teacher]-rudi_rules`.



## Dependencies

Our codes work perfectly with the followling packages:

- python=3.8.3
- pandas==1.2.3 (**Important: other versions of pandas are likely to raise unexcepted errors**)
- numpy==1.21.2
- pytorch==1.6.0
- scikit-learn==0.23.2
- scipy==1.7.3
- tqmd==4.48.2


## Datasets

Raw datasets can be downloaded from the following links.

- VEWS: https://cs.stanford.edu/~srijan/vews/vews-raw-dataset.zip
- Elo: https://www.kaggle.com/c/elo-merchant-category-recommendation/
- RedHat: https://www.kaggle.com/c/predicting-red-hat-business-value/

The pre-processing are included in `data_preprocessing`.
