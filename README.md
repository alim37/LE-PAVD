## Model Training

To run an individual training experiment, use:

```
cd /model/
python3 train.py {path to cfg} {path to dataset} {name of experiment} {dataset split}
```


## Model Evaluation

To evaluate an individual model, use:

```
cd /model/
python3 evaluate.py {path to cfg} {path to dataset} {path to model weights}
```
