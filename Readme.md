This is a Facial Expression Recognition Convolutional Neural Network model that is improved from WuJie's( WuJie1010 /Facial-Expression-Recognition.Pytorch Public) that is the most starred on Github. It enforces orthogonality on the kernel matrix(samaonline/Orthogonal-Convolutional-Neural-Networks), in order to achieve more diverse features and improve detection accuracy. 

The experiment on FER2013 DataSet shows its superiority than the original WuJie's, which verifies the hypothesis that orthogonality could improve the model's performance. For further understanding about the orthognality, it is introduces in the AVI2022 paper(https://dl.acm.org/doi/abs/10.1145/3531073.3534470) 

The code now runs on FER2013 dataset, use Resnet-18 network, output is categorial emotion labels, and in non-resume performing state(i.e., not resume from former checkpoint, but start from 0). The entry code for train the model is "mainpro_FER.py", the dataset is under "/data/fer2013.csv", since the csv file is 200+ MB, larger than 25MB limit, it is compressed in custom mode splitting it into multiple sub-compressed files, and when it is to be used, just unzip one of them will recover the whole csv file. It runs about 6 minutes for 1 epho(consist of train, public test, and private test). Thus, for 250 epho setting in the code, it runs about 24 hours(1 wholy day) for training a FER model. The code could be reproduced after downloaded and necessary libraries installed.

The test code to recognize an face image's emotion is "visualize.py", which is easy to understand.

Wish it helps.

