## Model Details

#### 14 observations (labels):
label_names = [ 'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']


3-Class model(0: negative, 1: positive, 2: uncertain):
https://arxiv.org/pdf/1901.07031.pdf

2-Class model(0: negative, 1: positive):
Choose the best from U-Zeros and U-Ones

U-Zeros model (0: negative, 1: positive, merge uncertain into negative for training):

U-Ones model (0: negative, 1: positive, merge uncertain into positive for training):
https://arxiv.org/abs/1705.02315  Wang, Xiaosong, Peng, Yifan, Lu, Le, Lu, Zhiyong, Bagheri, Mohammadhadi, and Summers, Ronald M. Chestx-ray8: Hospital-scale chest x-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases. arXiv preprint arXiv:1705.02315, 2017.


#### Input:
224x224 image, convert to RGB, random horizontal flip, normalized based on the mean and standard deviation of training dataset of ImageNet


#### CNN Model:
densenet121 https://arxiv.org/abs/1608.06993
initialize parameters from the model pre-trained on ImageNet:
http://www.image-net.org/papers/imagenet_cvpr09.pdf 

Bottleneck Features:  1x1024 

——————————————————————————————
#### 3-Class Output:
dense layer: 14x3,  {p_0, p_1, p_2} on each label,  without Softmax(), since we use the loss function CrossEntropyLoss()

Loss Function (14-label, 3-class):
for 3 classes on each label, we use CrossEntropyLoss(), which includes Softmax(), Log() and NLLLoss(). Then we take the average over 14 labels.

Final Output: apply Softmax() on only {p_0, p_1}, then use p_1 as the output of each label.
——————————————————————————————
#### 2-Class Output
dense layer: 14x1,  only {p_1} on each label,  without Sigmoid(), since we use the loss function BCEWithLogitsLoss()

Loss Function (14-label, 2-class):
we use BCEWithLogitsLoss(), which includes Sigmoid() and BCELoss(). Then we take the average over 14 labels.

Final Output:  apply Sigmoid() on {p_1}
——————————————————————————————


#### Optimizer
Adam: β1 = 0.9 and β2 = 0.999 as default
Learning rate: 1E-4
Decayed Factor: 10 / 2 epoch
Epoch Number: 6 or 4

#### Batch
Batch Size (based on the size of memory)
32 for 224x224, 16 for 320x320

#### Training Time
for 224x224: ~0.6 hour / epoch
for 320x320: ~1.3 hour / epoch


#### ROC and PR in Valid dataset
use 2-class {p_0, p_1}, there is no uncertain,
we output ROC and PR for 14 observations


### Some AUC(ROC) Comparison:



I hate the table in markdown
lets try

| fk | omg |
| -- | -- |
| fk | omg |


| wtf | wtf  |
| -- | -- |
| wtf | wtf  |



Type             | CheXNet |    CheXNet         |   CheXpert         |  CheXpert          | Ours    | Ours   | Ours    |
Name             | No U    |    No U            |   U-Ones           |  3-Class           | U-Zeros | U-Ones | 3-class |
---------------- | ------- | ------------------ | ------------------ | ------------------ | ------- | ------ | ------- |
Atelectasis      | 0.8094  | 0.862(0.825–0.895) | 0.858(0.806,0.910) | 0.821(0.763,0.879) | 0.75    | 0.81   | 0.75    |
Cardiomegaly     | 0.9248  | 0.831(0.790–0.870) | 0.832(0.773,0.890) | 0.854(0.800,0.909) | 0.84    | 0.79   | 0.85    |
Consolidation    | 0.7901  | 0.893(0.859-0.924) | 0.899(0.854,0.944) | 0.937(0.905,0.969) | 0.86    | 0.86   | 0.87    |
Edema            | 0.8878  | 0.924(0.886-0.955) | 0.941(0.903,0.980) | 0.928(0.887,0.968) | 0.93    | 0.93   | 0.93    |
Pleural Effusion | 0.8638  | 0.901(0.868-0.930) | 0.934(0.901,0.967) | 0.936(0.904,0.967) | 0.92    | 0.92   | 0.91    |
Pleural Other    | 0.8062  | 0.798(0.744-0.849) | NaN                | NaN                | 0.96    | 0.87   | 0.93	   |	
Pneumonia        | 0.7680  | 0.851(0.781-0.911) | NaN                | NaN                | 0.73    | 0.70   | 0.78    |	
Pneumothorax     | 0.8887  | 0.944(0.915-0.969) | NaN                | NaN                | 0.91    | 0.89   | 0.87    |


### Some AUC(PR) Comparison:
Type			|CheXpert	|Ours	|Ours	|Ours	|
Name			|Ensemble	|U-Zeros|U-Ones	|3-Class|
----------------|-----------|-------|-------|-------|
Atelectasis		|0.69		|		|0.38	|0.38	|
Cardiomegaly	|0.81		|		|0.56	|0.57	|
Consolidation	|0.44		|		|0.17	|0.21	|
Edema			|0.66		|		|0.68	|0.67	|
Pleural Effution|0.91		|		|0.86	|0.86	|


## Implementation

### Step 1
in ./

conda env create environment.yml

conda activate chexpert

### Step 2
in ./data/

unzip CheXpert-v1.0-small.zip, then ./data/ should be like this:

./data/ train/

		valid/

		train.csv

		valid.csv

modify the data path in datasplit.py, train.py if you need


### Step 3
in ./code/

run train.py, model will be saved in ./output/

### Step 4
If you just want to make the ROC, PR graph:
modify "transforms" in roc.py to make it consistant with your model
move model.pth into ./output/
in ./code/
run roc.py

