# Instructions for winning method in MICCAI 2017 WMH segmentation challenge

### testing your cases
A easy-to-use demo code could be downloaded here: https://drive.google.com/file/d/1tjk8CXjGYeddbaPCc1P5r-_ACUFcMut4/view?usp=sharing . It support single modality (FLAIR) and two-modality (FLAIR and T1) as the the input. The detailed instruction is in **ReadMe** inside. Please have a look at it.
Simply, just run: 
```
python test_your_data.py
```


### some instructions for the public codes
* Requirements: 
Keras 2.0.5, Tensorflow, Python 2.7, h5py


For the weights of the model we submitted to the challenge, please download them via: https://drive.google.com/drive/folders/1i4Y9M0yW3JN_WC8Fj1VlCdaE2lvG_9Ar . You can use these models for segmenting your cases. We also have Docker file to do segmentation if you are interested. Please feel free to contact me.   

For the .npy files to run the leave-one-subject-out experiments, please download via: https://drive.google.com/open?id=1m0H9vbFV8yijvuTsAqRAUQGGitanNw_k . This is the preprocessed data we used both the challenge and our NeuroImage paper. The preprocessing steps can be found in testing code. We followed the same procedures. The number of slices of each are reduced a bit by removing the first and last few slices, i.e., the num of slices in each subject in Utrecht were reduced to 38, in Amsterdam were 38 and GE3T were 63. The order of the subject were result generated by reading all the dir name in each subset and performing dir.sort(). For example the order in Utrecht should be: 0, 11, 17, 19, 2, 21...
So the structure is like this: Utrecht = data[0:760, ...], Amsterdam = data[760:1520, ...], GE3T = data[1520:2780, ...]


Decriptions for the python code:

train_leave_one_out.py: train U-Net models under leave-one-subject-out protocol. For options, you can train models with single modelity or without data augmentation.
test_leave_one_out.py: test U-Net models under leave-one-subject-out protocol. The codes also include the preprocessing of the original data.
evaluation.py: evaluation code provided by the challenge organizor. 

images_three_datasets_sorted.npy: preprocessed dataset including Utrecht, Singapore and GE3T. The order of the patients is sorted.
masks_three_datasets_sorted.npy: preprocessed masks including Utrecht, Singapore and GE3T corresponding to the preprocessed data. The order of the patients is sorted.



* The detailed description of our method is published in NeuroImage: https://www.sciencedirect.com/science/article/pii/S1053811918305974?via%3Dihub .
* Please cite our work if you find the code is useful for your research.

