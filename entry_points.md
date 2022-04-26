Different commands for performing the task

i. Data augmentation => 
python augmentation.py data/train.csv

ii. Data cleaning => 
Training data:
python clean_data.py data/updated/augmented.csv data/updated/train_cleaned.csv
Validation data:
python clean_data.py data/validation.csv data/updated/val_cleaned.csv
Testing data:
python clean_data.py data/test.csv data/updated/test_cleaned.csv

iii. Training =>
python train.py data/train.csv data/updated/train_cleaned.csv data/updated/val_cleaned.csv

iv. Predict and output solution.csv =>
python predict.py data/train.csv data/updated/train_cleaned.csv data/updated/test_cleaned.csv