# Knowledge Tracing Collection with PyTorch

This repository is a collection of the following knowledge tracing algorithms:
- **Deep Knowledge Tracing (DKT)**

## Install Dependencies 
1. Install Python 3.
2. <Optional> Create Conda environment

    ```bash
    $ conda create -n dkt python=3.9
    $ conda activate dkt
    ```
3. Install the Python packages in `requirements.txt`. If you are using a virtual environment for Python package management, you can install all python packages needed by using the following bash command:

    ```bash
    $ pip install -r requirements.txt
    ```
4. Install PyTorch from https://pytorch.org/get-started/locally/

    ```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```


## Training
    ```bash
    $ python train.py --model_name=dkt --dataset_name=ASSIST2009
    ```
    
    model_name options: dkt, dkt+, dkvmn, sakt, gkt(very slow)
    dataset_name options: ASSIST2009, ASSIST2015, Algebra2005

    The following bash command will help you:
    ```bash
    $ python train.py -h
    ```

## Inference
    ```bash
    $ python predict.py --model_name=dkt --dataset_name=ASSIST2009 --student_hist_path=student001.csv
    ```

    student_hist_path should lead to a CSV file which records the past exercise activities (skill name + correct or not)
    
    Outputs in student_hist_path_report.csv show the probability of getting each skill correct after the input activities

