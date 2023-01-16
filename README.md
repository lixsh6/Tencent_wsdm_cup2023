# WSDM Cup 2023 Source Code:

-------------
- This repo contains the source code of our competition in WSDM Cup 2023: [Pre-training for Web Search](https://aistudio.baidu.com/aistudio/competition/detail/536/0/leaderboard) and [Unbiased Learning for Web Search](https://aistudio.baidu.com/aistudio/competition/detail/534/0/leaderboard).
- In the Pre-training task, we implement all codes in both **Pytorch** and **PaddlePaddle** version (You can **pretrain & finetune** in anyone of these two frameworks.).   
- In the Unbiased LTR task, we implement codes in **Pytorch** version. 

**Below is details for Pre-training task. For Unbiased LTR task, see [README.md](./pytorch_unbias/README.md) for details.**
## Quick Links
- [Method Overview](#method-overview)
- [Training](#training)
- [Ensemble learning](#ensemble-learning)
- [Reproduce results on leaderboard](#reproduce-results-on-leaderboard)
- [Environment](#environment)

## Method Overview
- Pre-training BERT with MLM and CTR prediction loss (or multi-task CTR prediction loss).
- Finetuning BERT with pairwise ranking loss.
- Obtain prediction scores from different BERTs.
- Ensemble learning to combine BERT features and sparse features.

Details will be updated in the submission paper.

## Training 
- In all `start.sh` files in `pytorch_pretrain (or paddle_pretrain)`, modify all `data_root={Palce Your Data Root Path Here}/baidu_ultr` as your file path.
- Set `NPROC` as your GPU number. 
### 1) Pretraining
```angular2html
cd pytorch_pretrain/pretrain (or paddle_pretrain/pretrain)
sh start.sh
```
### 2) Finetuning
```angular2html
cd pytorch_pretrain/finetune (or paddle_pretrain/finetune)
sh start.sh
```
### 3) Inference for submission
```angular2html
cd pytorch_pretrain/submit (or paddle_pretrain/submit)
sh start.sh
```
## Ensemble learning
We use [lambdamart](https://medium.datadriveninvestor.com/a-practical-guide-to-lambdamart-in-lightgbm-f16a57864f6) by [lightgbm](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRanker.html) to ensemble different scores from the finetuned bert models. 

#### Sparse features:
- query length
- document length
- query frequency 
- number of hit words of query in document
- BM25 score
- TF-IDF score

#### BERT features:
##### 1) Model details: [Download](https://huggingface.co/lixsh6/wsdm23_pretrain/tree/main)
| Index| Model Flag    | Method | Pretrain step | Finetune step | DCG on leaderboard | 
| --------| -------- | ------- |---------------| ------- | ------- | 
| 1| large_group2_wwm_from_unw4625K | M1 | 1700K         | 5130 | 11.96214 |
| 2| large_group2_wwm_from_unw4625K | M1 | 1700K         | 5130 | NAN |
| 3| base_group2_wwm | M2 | 2150K         | 5130 | ~11.32363 |
| 4| large_group2_wwm_from_unw4625K | M1 | 590K          | 5130 | 11.94845 |
| 5| large_group2_wwm_from_unw4625K | M1 | 1700K         | 4180 | NAN |
| 6| large_group2_mt_pretrain | M3 | 1940K         | 5130 | NAN |

##### 2) Method details

| Method  | Model Layers |   Details |
| -------- | ------- | ------- |
| M1 | 24 | WWM & CTR prediction as pretraining tasks|
| M2 | 12 | WWM & CTR prediction as pretraining tasks |
| M3 | 24 | WWM & Multi-task CTR prediction as pretraining tasks|

#### The procedure contains two steps:
1. Cross validation on validation set to determine best parameters. See `./lambdamart/cross_validation.ipynb`.
2. Generate the final scores based on the determined parameters in step 1. See `./lambdamart/run.ipynb`.

## Reproduce results on leaderboard
### 1) Convert Torch checkpoint to Paddle checkpoint.
- Install [X2Paddle](https://github.com/PaddlePaddle/X2Paddle) and [onnxsim](https://pypi.org/project/onnxsim/)
- We use method one in this [link](https://github.com/PaddlePaddle/X2Paddle/blob/develop/docs/inference_model_convertor/pytorch2paddle.md). Using the following three commands:
```
python ./paddle_pretrain/convert/convert-onnx.py 
python -m onnxsim model.onnx model_sim.onnx
x2paddle --framework=onnx --model=model_sim.onnx --save_dir=./pd_model
```

It will output a folder named `./pd_model` which contains `x2paddle.py`(Model definition in paddlepaddle) and `model.pdparams` (Trained parameters). Copy `x2paddle.py` to `./paddle_pretrain/review/x2paddle.py`. We already generate a file there which converts a 24-layer model, you can directly use this file if your always use a 24-layer model. 

### 2) Inference score for each bert model
- Modify `data_root` as your path in `./paddle_pretrain/review/start.sh` and then run it by `sh start.sh`.
- It uses PaddlePaddle framework to inference the score of each query-document pair.

### 3) Ensemble learning
- We already inference scores of 6 different models. The Scores are all contained in `./lambdamart/features`.
- Run all cells in `./lambdamart/run.ipynb`. It will reproduce the scores of our final submission by ensembling all scores from different models, which is the same as `./lambdamart/features/final_result_submit.csv`.

## Environment

We opensource dockers for both pytorch and paddlepaddle to save your configuration time of environment.

| Version      | Key configuration                                                      |
|--------------|------------------------------------------------------------------------| 
| Pytorch      | <ul><li>Python 3.6</li><li>torch1.8.0</li><li>transformers-4.18.0</ul> | 
| PaddlePaddle | <ul><li>Python 3.9</li><li>Paddle2.4</li><li>cuda11.2-cudnn8.2</ul>  | 
To be updated.

## Contacts
- Xiangsheng Li: [lixsh6@gmail.com](lixsh6@gmail.com).
- Xiaoshu Chen:  [xschenranker@gmail.com](xschenranker@gmail.com)
