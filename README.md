# White Matter Hyperintensity segmentation
A white matter hyperintensity segmentation tool based on sysu_media, the winnner of the MICCAI 2017 WMH challenge. A range of updates to functionailty and some tweaks in methodolgy have been performed to improve the method. 

## Installation 
The code has been tested using python 3.6.4 but should work for most python 3.6+ versions. The main dependencies of this method are, Keras 2.3.1, TensorFlow 1.14 and SimpleITK 2.0.1 among others. Installation of the dependencies should be as simple as

``` bash
pip3 install -r requirements.txt
```


## Usage
The body of the method is in the [wmh](wmh) folder, which in turn contains the [weights](wmh/weights) folder which contains the weights of the pre-trained networks. 

The main scripts for using the code are:
    - [eval.py](eval.py) which evaluates a pre-trained network on a given dataset. *This is the script to use for prediction*
    - [train.py](train.py) which trains a network either from scratch, or given starting weights
    - [random_test_train_split.py](random_test_train_split.py) which, given a directory full of subjects, will split them into training and testing sets and produce csv files listing subjects in each group
    - [generate_hdf5.py](generate_hdf5.py) which takes a list of subjects (either csv from random_test_train_split.py or pointed to directory) and loads in images, preprocesses and saves into a .h5 file ready for training.
    
### Prediction
An example code to evaluate model given a .csv file from random_test_train_split

``` bash
python3 ./eval.py --csv_file ./subjects_ADNI_train.csv --num_unet 3 --compute_metrics --T1_name mri/brain-in-rawavg_mgz2nii.nii.gz --FLAIR_name flair.nii.gz --gt_name mri/wmh.nii.gz
```

<details>
  <summary>
    Full usage
  </summary>  

```
usage: eval.py [-h] [--data_dir DATA_DIR] [--csv_file CSV_FILE]
               [--pattern PATTERN] [--T1_name T1_NAME]
               [--FLAIR_name FLAIR_NAME] [--gt_name GT_NAME]
               [--output_name OUTPUT_NAME] [--rows_standard ROWS_STANDARD]
               [--cols_standard COLS_STANDARD] [--batch_size N] [--verbose]
               [--model_dir MODEL_DIR] [--FLAIR_only] [--num_unet NUM_UNET]
               [--num_unet_start NUM_UNET_START] [--ignore_frac IGNORE_FRAC]
               [--compute_metrics] [--model_suffix MODEL_SUFFIX]

WMH training

optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   Directory containing data. Will be overrided by
                        --csv_file is supplied
  --csv_file CSV_FILE   Csv-file listing subjects to include in file
  --pattern PATTERN     Pattern to match files in directory.
  --T1_name T1_NAME     Default name of T1 images. (default
                        T1/T1_brain.nii.gz)
  --FLAIR_name FLAIR_NAME
                        Default name of T2FLAIR images. (default
                        T2_FLAIR/T2_FLAIR)
  --gt_name GT_NAME     Default name for ground truth segmentations (default
                        T2_FLAIR/lesions/final_mask.nii.gz)
  --output_name OUTPUT_NAME
                        Name of ouput segmentation file. (default
                        wmh_seg.nii.gz)
  --rows_standard ROWS_STANDARD
                        Height of input to network (Default 200)
  --cols_standard COLS_STANDARD
                        Width of input to network (Default 200)
  --batch_size N        input batch size for training (default: 30)
  --verbose, -v         Flag to use verbose training. A single flag will cause
                        full verbosity. Double flag (e.g. -vv) will cause less
                        verbosity (use -vv in non-interactive environments
                        like cluster)
  --model_dir MODEL_DIR
                        path to store model weights to (also path containing
                        starting weights for --resume) (default:
                        ./wmh/weights)
  --FLAIR_only          Flag whether to just use FLAIR (default (if flag not
                        provided): use FLAIR and T1)
  --num_unet NUM_UNET   Number of networks to train (default: 1)
  --num_unet_start NUM_UNET_START
                        Number from which to start training networks (i.e.
                        start from network 1 if network 0 is done) (default:
                        0)
  --ignore_frac IGNORE_FRAC
                        Fraction of slices from top and bottome to ignore
                        (default: 0.125)
  --compute_metrics     Flag whether to compute metrics after segmentation
                        (requires ground truth)
  --model_suffix MODEL_SUFFIX
                        Suffix to model name so model will save as
                        {num_net}_{suffix}.h5 (default: None)

```
</details>

### Training 
For training, first a dataset must be prepared using generate_hdf5.py and/or random_test_train_split.py. For example, this is how one could do all the steps necesary to train a model from scratch on a dataset:

``` bash
> python3 ./random_test_train_split.py --data_dir ../ADNI_data --csv_name ./subjects_ADNI
> python3 ./generate_hdf5.py --csv_file ./subjects_ADNI_train.csv --hdf5_name ADNI_train_data.hdf5 --T1_name mri/T1.nii.gz --FLAIR_name flair.nii.gz --gt_name mri/wmh.nii.gz 
> python3 ./train.py --batch_size 32 --verbose --hdf5_name_train ADNI_train_data.hdf5
```

<details>
<summary>
Full usage for train.py
</summary>

```
usage: train.py [-h] [--hdf5_name_train HDF5_NAME_TRAIN]
                [--hdf5_name_test HDF5_NAME_TEST]
                [--validation_split VALIDATION_SPLIT] [--batch_size N]
                [--validation_batch_size N] [--epochs EPOCHS] [--verbose]
                [--early_stopping] [--es_patience ES_PATIENCE]
                [--es_metric {loss,dice,dsc,jaccard,tversky,focal-tversky}]
                [--log_dir LOG_DIR] [--csv_log] [--tb_log]
                [--model_dir MODEL_DIR]
                [--loss {dice,jaccard,dsc,tversky,focal-tversky}]
                [--metrics [{dice,jaccard,dsc,tversky,focal-tversky} [{dice,jaccard,dsc,tversky,focal-tversky} ...]]]
                [--resume] [--FLAIR_only] [--no_aug] [--aug_factor AUG_FACTOR]
                [--aug_theta AUG_THETA] [--aug_shear AUG_SHEAR]
                [--aug_scale AUG_SCALE] [--num_unet NUM_UNET]
                [--num_unet_start NUM_UNET_START] [--test_ensemble] [--lr LR]
                [--output_test_aug] [--no_shuffle]

WMH training

optional arguments:
  -h, --help            show this help message and exit
  --hdf5_name_train HDF5_NAME_TRAIN
                        path and name of hdf5-dataset for training (default:
                        test_train.hdf5)
  --hdf5_name_test HDF5_NAME_TEST
                        path and name of hdf5-dataset for testing (default:
                        test_test.hdf5)
  --validation_split VALIDATION_SPLIT
                        Fraction of data for validation. Will be overridden by
                        hdf5_name_test for explicit validation set. (default:
                        0.2)
  --batch_size N        input batch size for training (default: 30)
  --validation_batch_size N
                        input batch size for validation (default: 30)
  --epochs EPOCHS       Number of epochs (default: 50)
  --verbose, -v         Flag to use verbose training output. -v will have
                        progress bar per epoch, -vv will print one line per
                        epoch (use this in non-interactive runs e.g. cluster)
  --early_stopping      Flag to use early stopping
  --es_patience ES_PATIENCE
                        No. epochs over which to use patience in early
                        stopping (default: 20)
  --es_metric {loss,dice,dsc,jaccard,tversky,focal-tversky}
                        Choice of early stopping monitoring metric (default:
                        loss)
  --log_dir LOG_DIR     Log directory for logging of training performance.
                        Requires --csv_log to be provided for logging
                        (default: None)
  --csv_log             Flag to store csv log
  --tb_log              Flag to store tensor board log
  --model_dir MODEL_DIR
                        path to store model weights to (also path containing
                        starting weights for --resume) (default:
                        ./wmh/weights)
  --loss {dice,jaccard,dsc,tversky,focal-tversky}
                        Choice of loss function (default: dice)
  --metrics [{dice,jaccard,dsc,tversky,focal-tversky} [{dice,jaccard,dsc,tversky,focal-tversky} ...]]
                        Choice of metric functions (default: None)
  --resume              Flag to resume training from checkpoints.
  --FLAIR_only          Flag whether to just use FLAIR (default (if flag not
                        provided): use FLAIR and T1)
  --no_aug              Flag to not do any augmentation
  --aug_factor AUG_FACTOR
                        Factor by which to increase dataset by using
                        augmentation. i.e. the dataset will be x times bigger
                        after augmentation (default: 1 (results in twice as
                        big a dataset))
  --aug_theta AUG_THETA
                        Degree of rotation to use in augmentation [degrees]
                        (default: 15)
  --aug_shear AUG_SHEAR
                        Shear factor in augmentation (default: 0.1)
  --aug_scale AUG_SCALE
                        Scaling factor in augmentation (default: 0.1)
  --num_unet NUM_UNET   Number of networks to train (default: 1)
  --num_unet_start NUM_UNET_START
                        Number from which to start training networks (i.e.
                        start from network 1 if network 0 is done) (default:
                        0)
  --test_ensemble       Flag to test the overall ensemble performance once all
                        networks are trained
  --lr LR               Learning rate (default: 2e-4)
  --output_test_aug     Flag to save 10 test images from augmentation
                        generator
  --no_shuffle          Flag to not shuffle the slices during training
                        (default is to shuffle)
```
</details>

### Citation
#### sysu_media
This segmentation code is based on the sysu_media code published in [NeuroImage](https://arxiv.org/pdf/1802.05203.pdf). Please cite our work if you find the codeis useful for your research.
