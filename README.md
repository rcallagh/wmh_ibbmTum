# White Matter Hyperintensity segmentation
A white matter hyperintensity segmentation tool based on sysu_media, the winnner of the MICCAI 2017 WMH challenge. A range of updates to functionailty and some tweaks in methodolgy have been performed to improve the method. 

## Installation 
The code has been tested using python 3.6.4 but should work for most python 3.6+ versions. The main dependencies of this method are, Keras 2.3.1, TensorFlow 1.14 and SimpleITK 2.0.1 among others. Installation of the dependencies should be as simple as
'''
pip3 install -r requirements.txt
'''

## Usage
The body of the method is in the [wmh](wmh) folder, which in turn contains the [weights](wmh/weights) folder which contains the weights of the pre-trained networks. 

The main scripts for using the code are:
    - [eval.py](eval.py) which evaluates a pre-trained network on a given dataset. *This is the script to use for prediction*
    - [train.py](train.py) which trains a network either from scratch, or given starting weights
    - [random_test_train_split.py](random_test_train_split.py) which, given a directory full of subjects, will split them into training and testing sets and produce csv files listing subjects in each group
    - [generate_hdf5.py](generate_hdf5.py) which takes a list of subjects (either csv from random_test_train_split.py or pointed to directory) and loads in images, preprocesses and saves into a .h5 file ready for training.
    
### Prediction
An example code to evaluate model given a .csv file from random_test_train_split
'''bash
python3 ./eval.py --csv_file ./subjects_ADNI_train.csv --num_unet 3 --compute_metrics --T1_name mri/brain-in-rawavg_mgz2nii.nii.gz --FLAIR_name flair.nii.gz --gt_name mri/wmh.nii.gz
'''
<details>
  <summary>
    Full usage
  </summary>  
'''
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
'''
</details>




### Citation
#### sysu_media
This segmentation code is based on the sysu_media code published in [NeuroImage](https://arxiv.org/pdf/1802.05203.pdf). Please cite our work if you find the codeis useful for your research.
