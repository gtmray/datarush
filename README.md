## DOCSUMO DATARUSH TEAM PDSC

# Installation

Create a virtual environment within the 'project folder', activate it, and run this command below.


```bash
pip install -r requirements.txt
```

# Files Description and Usage
### augmentation.py -> Create data augmentation from original file
```bash
python augmentation.py ORIGINAL_TRAIN_FILE_PATH
```
### clean_data.py -> Data cleaning
```bash
python clean_data.py SOURCE_PATH DESTINATION_PATH
```
#### NOTE: SOURCE_PATH must be an augmented file for training data
### train.py -> Training script
```
python train.py ORIGINAL_TRAIN_FILE_PATH SRC_TRAIN SRC_VAL
```
#### NOTE: SRC_TRAIN AND SRC_VAL must be cleaned with clean_data.csv

### predict.py -> Predict and output solution.csv
```
python predict.py ORIGINAL_TRAIN_FILE_PATH SRC_TRAIN SRC_TEST
```
#### NOTE: SRC_TRAIN AND SRC_TEST must be cleaned with clean_data.csv

==========================================================


# Different commands for performing the task

- ### Data augmentation 
```bash
python augmentation.py data/train.csv
```

- ### Data cleaning 

 - #### Training data cleaning
```bash
python clean_data.py data/updated/augmented.csv data/updated/train_cleaned.csv
```
   - #### Validation data cleaning
```bash
python clean_data.py data/validation.csv data/updated/val_cleaned.csv
```
- #### Testing data cleaning
```bash
python clean_data.py data/test.csv data/updated/test_cleaned.csv
```

- ### Training
```bash
python train.py data/train.csv data/updated/train_cleaned.csv data/updated/val_cleaned.csv
```
- ### Predict and output solution.csv
```bash
python predict.py data/train.csv data/updated/train_cleaned.csv data/updated/test_cleaned.csv
```