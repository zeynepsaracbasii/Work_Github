# Turkish Text Classification with BERT

In this project, BERT model which is fine-tuned of https://github.com/stefan-it/turkish-bert  is used for Turkish Text Classification.


## Data
                
For this classifictaion task, a new dataset was created with fake data. The Fake Dataset consists of 6 classes.

Phone number
Republic of Turkey Identity Number
Hobbies
Amount
Credit Card Number
Blood type

## Model
```
from simpletransformers.classification import ClassificationModel

model = ClassificationModel(
    "bert", 
    "dbmdz/bert-base-turkish-cased",
    num_labels=6,
    use_cuda=True, 
    args={'reprocess_input_data': True, 
          'overwrite_output_dir': True, 
          'num_train_epochs': 3,
          'use_early_stopping': True})
```
## Make Prediction with Model
To predict some text model_prediction() function is created.

``` 
model_prediction("su sporları")
'Hobbies'
```
``` 
model_prediction("28973456 8764 2613")
'Credit Card Number'
```
``` 
model_prediction("A Positive")
'Blood type'
```
