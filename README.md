# asynchronous-federated-learning
This is the repository for the work "An asynchronous aggregation mechanism with normalized temporal and informative weights and optimized interaction patterns"


##requirements
python
tensorflow
keras


##run
1.[Generate Local Data Sets-LN.py]
 >>>Generate non-IID data set saved in [non-iid_data_mnist_range5_label].

2.[mnist_cnn.py]
>>>Generate initialized model saved in [w0.h5] and see local model structure in detail.

3.[FedAvg.py]
>>>Federated averaging algorithm.

4.[MLU.py]
>>> Employ multi-phase layer updating strategy (MLU).

5.[TWF.py]
>>> Employ temporal weight fading strategy (TWF).

6.[IWE-IE.py] [IWE-LN]
>>> Employ informative weight enhancing strategy (IWE), include information entropy (IE) and label number (LN).

7.[MLU+IWE-IE.py] [MLU+IWE-LN]
>>> Employ MLU+IWE.

8.[MLU+TWF.py]
>>> Employ MLU+TWF.

9.[MLU+TWF+IWE-IE.py] [MLU+TWF+IWE-LN.py]
>>>Employ MLU+TWF+IWE.


##results
![image](https://user-images.githubusercontent.com/73324626/120967236-d60c8300-c799-11eb-96f9-b6f7a2c76b86.png)
