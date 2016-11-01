# tf-char-cnn-lstm
A Tensorflow version of Yoon Kim's char-cnn-lstm [Torch7 code](https://github.com/yoonkim/lstm-char-cnn)

It is a boosting and amended version of [mkroutikov's work](https://github.com/mkroutikov/tf-lstm-char-cnn),to fix some mistakes and improve the speed.Now the training time is within 4 hours on large model, much faster than the original 20+ hours.And we can get better result at the same time.

A simplified and neaty version called **train_sim.py** is also provided for beginners,aimed to make the whole process easy to understand.It is identical to the train.py, but in a straight-forward style. 

## requirement

Tensorflow 0.10
cuda and cudnn should be installed for gpu implement

## usage
Train:
```sh
python train_gpu.py
```
Evaluate
```sh
python evaluate_gpu.py
```
### for CPU:
Train：
```sh
python train_gpu.py --gpuid -1
```
Evaluate:
```sh
python evaluate_gpu.py --gpuid -1
```
Large model of Yoon Kim's paper will be trained on PTB and also evaluated.

## Time
The training time the large model is about 3.5~3.8 hour on a GPU(k20),while the lua code of Yoon Kim is about 5 hours on a GPU.


## Result
| Learning rate  |  Train/Valid/Test loss  |  Train/Valid/Test perplexity  |
|:--------------:|:-----------------------:|:------------------------------|
| 1.0            | 4.057 / 4.503 / 4.463   | 57.77 / 90.25 / 86.79         |
| 0.5            | 3.984 / 4.432 / 4.391   | 53.71 / 84.06 / **80.73**     |

The best result is 80.73 of PPL,which is similar to the **79** of Yoon Kim's result.






