This is a Facial Expression Recognition Convolutional Neural Network model that is improved from WuJie's( WuJie1010 /Facial-Expression-Recognition.Pytorch Public) that is the most starred on Github. It enforces orthogonality on the kernel matrix(samaonline/Orthogonal-Convolutional-Neural-Networks), in order to achieve more diverse features and improve detection accuracy. 

The experiment on FER2013 DataSet shows its superiority than the original WuJie's, which verifies the hypothesis that orthogonality could improve the model's performance. For further understanding about the orthognality, it is introduces in the AVI2022 paper(https://dl.acm.org/doi/abs/10.1145/3531073.3534470) 

The code now runs on FER2013 dataset, use Resnet-18 network, and in non-resume performing state(i.e., not resume from former checkpoint, but start from 0). The entry code is "mainpro_FER.py", the dataset is under "/data/fer2013.csv", since the csv file is 200+ MB, larger than 25MB limit, it is compressed in custom mode splitting it into multiple sub-compressed files, and when it is to be used, just unzip one of them will recover the whole csv file.

The code could be reproduced after downloaded and necessary libraries installed. Wish it helps.

