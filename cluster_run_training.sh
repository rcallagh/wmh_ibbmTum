#!/bin/bash
#====================
# QSUB
#====================
#$ -l h_rt=4:00:0
#$ -l tmem=20G
#$ -l gpu=true
#$ -N whm_train
#$ -S /bin/bash
#$ -j y
#$ -o /SAN/medic/camino_2point0/Ross/wmh/
#$ -t 1-3

#===== PYTHON/CUDA setup ============================
source /share/apps/source_files/python/python-3.6.4.source
source /share/apps/source_files/cuda/cuda-10.0.source
#===================================================

#Setup directories for WMH seg code, python venv
WMH_FOLDER="/SAN/medic/camino_2point0/Ross/wmh/"
PY_VENV_FOLDER="/home/rcallagh/py_venv/wmh/"

#Location of training and testing data
train_hdf="/SAN/medic/camino_2point0/Ross/wmh/biobank_train_data.hdf5"
test_hdf="/SAN/medic/camino_2point0/Ross/wmh/biobank_test_data.hdf5"

#Network params
num_unet=1
num_unet_start=$((SGE_TASK_ID-1))

# Create folder for python venv if does not exist already
if [ ! -d ${PY_VENV_FOLDER} ]; then
mkdir $PY_VENV_FOLDER
fi
# Create python virtual environment if does not exist
if [ -f ${PY_VENV_FOLDER}/bin/activate ]; then
        echo "Sourcing ${PY_VENV_FOLDER}/bin/activate ..."
        source ${PY_VENV_FOLDER}/bin/activate
else
        echo "Creating virtual environment in ${PY_VENV_FOLDER}"
        python3 -m venv ${PY_VENV_FOLDER}
        source ${PY_VENV_FOLDER}/bin/activate
        pip install -r ${FAST_SURFER_FOLDER}/requirements.txt
fi


date
hostname
# Change directory to the main FS
#cd ${WMH_FOLDER}
# Define the three planes
# Define the outputs
input_train="${out_dir_train}/training_set_${pln}.hdf5"
input_valid="${out_dir_valid}/validation_set_${pln}.hdf5"
# Assumes to restart the training from the provided checkpoints
# First letter of the plane is capital
cap_pln="$(tr '[:lower:]' '[:upper:]' <<< ${pln:0:1})${pln:1}"
log_dir="../checkpoints/${cap_pln}_Weights_FastSurferCNN"
pwd
# Run the python function for training set
python "${WMH_FOLDER}train.py" \
       --hdf5_name_train ${train_hdf} \
       --batch_size 30 \
       --epochs 60 \
       --resume \
       --num_unet ${num_unet} \
       --num_unet_start ${num_unet_start} \
       --model_dir "${WMH_FOLDER}/wmh/weights/" \
       -vv


date
