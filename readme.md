End-to-end Learning for Early Classification of Time Series (ELECTS)
===

Execute hyperparameter tuning and training
```
python main.py --model Conv1D --dataset GunPoint
```

Execute single run
```bash
train.py -d GunPoint -m Conv1D --store /tmp --batchsize 256 --overwrite
```


```bash
train.py -d BavarianCrops_uniform_2500 -m DualOutputRNN --loss-mode "early_reward" --store /tmp --batchsize 256 --overwrite
```

<img width=200px src="docs/conv1d.png"/>

### Runs (visdom)

Gunpoint

<img width=100% src="docs/GunPoint_run.png"/>

Wafer

<img width=100% src="docs/Wafer_run.png"/>

EGC

<img width=100% src="docs/EGC_run.png"/>

Remote Sensing Dataset

<img width=100% src="docs/visdom_bavariancrops.png"/>



### Download data

```bash
wget https://corupublic.s3.eu-central-1.amazonaws.com/BavarianCropsHoll8.zip /data/
unzip /data/BavarianCropsHoll8.zip -d /data/
```

```bash
wget https://s3.eu-central-1.amazonaws.com/corupublic/early_rnn.zip
```