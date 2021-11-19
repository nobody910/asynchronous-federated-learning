# asynchronous-federated-learning

This is the repository for the work "An ensemble mechanism to tackle the heterogeneity in asynchronous federated learning"


##requirements

python
tensorflow
keras


##run

1.[Generate Local Data Sets mnist.py]
 >>>Generate non-IID data set saved in [data_mnist_noniid].

2.[mnist_cnn.py]
>>>Generate initialized model saved in [w0.h5] and see local model structure in detail.

3.[FedAvg_mnist.py]
>>>Federated averaging algorithm.

4.[AMU_rdm_mnist.py]
>>> Employ adaptive model updating strategy (AMU).

5.[AMU+TVW+IDW-IE_mnist.py] [AMU+TVW+IDW-LN_mnist.py]
>>>Employ AMU + TVW(time variety weighted strategy) + IDW( information diversity weighted strategy (IDW)).

