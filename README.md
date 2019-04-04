** Step 1
in ./
conda env create environment.yml
conda activate chexpert

** Step 2
in ./data/
unzip CheXpert-v1.0-small.zip, then ./data/ should be like this:
./data/ train/
		valid/
		train.csv
		valid.csv

** Step 3
in ./code/
run datasplit.py, add train, valid, test datasets into ./data/:
./data/ train/
		valid/
		train.csv
		valid.csv
		data_train.csv
		data_valid.csv
		data_test.csv

it also fillna(0) for 14 labels 

** Step 4
in ./code/
modify the data path in train.py if not the same with above
run train.py


