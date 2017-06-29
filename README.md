# Bi-directional Attention Flow for Machine Comprehension
 
- This is **originally** forked from https://github.com/allenai/bi-att-flow
- Support **Tensorflow 1.2.0 (>=1.0.0)** and clean up the code.
- **Pretrained weights** are provided
- **Demo** is integrated into master branch
- =================================
- This the original implementation of [Bi-directional Attention Flow for Machine Comprehension][paper].
- The CodaLab worksheet for the [SQuAD Leaderboard][squad] submission is available [here][worksheet].
- Please contact [Minjoon Seo][minjoon] ([@seominjoon][minjoon-github]) for questions and suggestions.

## 0. Requirements
#### General
- Python (developed on 3.5.2. Issues have been reported with Python 2!)
- unzip

#### Python Packages
- tensorflow (deep learning library, verified on 1.2.0)
- nltk (NLP tools, verified on 3.2.1)
- tqdm (progress bar, verified on 4.7.4)
- jinja2 (for visaulization; if you only train and test, not needed)
- install the dependencies via `pip install -r requirements.txt`

## 1. Pre-processing
First, prepare data. Donwload SQuAD data and GloVe and nltk corpus
(~850 MB, this will download files to `$HOME/data`):
```
chmod +x download.sh; ./download.sh
```

Second, Preprocess Stanford QA dataset (along with GloVe vectors) and save them in `$PWD/data/squad` (~5 minutes):
```
python -m squad.prepro
```

## 2. Training
The model has ~2.5M parameters.
The model was trained with NVidia Titan X (Pascal Architecture, 2016).
The model requires at least 12GB of GPU RAM.
If your GPU RAM is smaller than 12GB, you can either decrease batch size (performance might degrade),
or you can use multi GPU (see below).
The training converges at ~18k steps, and it took ~4s per step (i.e. ~20 hours).

Before training, it is recommended to first try the following code to verify everything is okay and memory is sufficient:
```
python -m basic.cli --mode train --noload --debug
```

Then to fully train, run:
```
python -m basic.cli --mode train --noload
```

You can speed up the training process with optimization flags:
```
python -m basic.cli --mode train --noload --len_opt --cluster
```
You can still omit them, but training will be much slower.

Note that during the training, the EM and F1 scores from the occasional evaluation are not the same with the score from official squad evaluation script. 
The printed scores are not official (our scoring scheme is a bit harsher).
To obtain the official number, use the official evaluator (copied in `squad` folder, `squad/evaluate-v1.1.py`). For more information See 3.Test.


## 3. Test
To test, run:
```
python -m basic.cli
```

Similarly to training, you can give the optimization flags to speed up test (5 minutes on dev data):
```
python -m basic.cli --len_opt --cluster
```

This command loads the most recently saved model during training and begins testing on the test data.
After the process ends, it prints F1 and EM scores, and also outputs a json file (`$PWD/out/basic/00/answer/test-####.json`,
where `####` is the step # that the model was saved).
Note that the printed scores are not official (our scoring scheme is a bit harsher).
To obtain the official number, use the official evaluator (copied in `squad` folder) and the output json file:

```
python squad/evaluate-v1.1.py $HOME/data/squad/dev-v1.1.json out/basic/00/answer/test-####.json
```

### 3.1 Loading from pre-trained weights

Instead of training the model yourself, you can choose to use **pre-trained weights**. Since they are not that big, I just put them in this repo. They are here `out/basic/00/save/basic-10000.*`

Note that these weights are not from the original author. The original author's pretrained weights are for old version of tensorflow, so I replaced them with mine.

For more details, please see `Demo`

### 4. Demo
To Run demo, just execute:
```
PYTHONPATH=. python3 demo/run_demo.py
```

## Results

### Dev Data

Note these scores are from the official evaluator (copied in `squad` folder, `squad/evaluate-v1.1.py`). For more information See 3.Test.
The scores appeared during the training could be lower than the scores from the official evaluator. 

|          | EM (%) | F1 (%) |
| -------- |:------:|:------:|
| single   | 67.7   | 77.3   |
| ensemble | 72.6   | 80.7   |

### Test Data

|          | EM (%) | F1 (%) |
| -------- |:------:|:------:|
| single   | 68.0   | 77.3   |
| ensemble | 73.3   | 81.1   |

Refer to [our paper][paper] for more details.
See [SQuAD Leaderboard][squad] to compare with other models.


<!--
## Using Pre-trained Model

If you would like to use pre-trained model, it's very easy! 
You can download the model weights [here][save] (make sure that its commit id matches the source code's).
Extract them and put them in `$PWD/out/basic/00/save` directory, with names unchanged.
Then do the testing again, but you need to specify the step # that you are loading from:
```
python -m basic.cli --mode test --batch_size 8 --eval_num_batches 0 --load_step ####
```
-->


## Multi-GPU Training & Testing
Our model supports multi-GPU training.
We follow the parallelization paradigm described in [TensorFlow Tutorial][multi-gpu].
In short, if you want to use batch size of 60 (default) but if you have 3 GPUs with 4GB of RAM,
then you initialize each GPU with batch size of 20, and combine the gradients on CPU.
This can be easily done by running:
```
python -m basic.cli --mode train --noload --num_gpus 3 --batch_size 20
```

Similarly, you can speed up your testing by:
```
python -m basic.cli --num_gpus 3 --batch_size 20 
``` 

[multi-gpu]: https://www.tensorflow.org/versions/r0.11/tutorials/deep_cnn/index.html#training-a-model-using-multiple-gpu-cards
[squad]: http://stanford-qa.com
[paper]: https://arxiv.org/abs/1611.01603
[worksheet]: https://worksheets.codalab.org/worksheets/0x37a9b8c44f6845c28866267ef941c89d/
[minjoon]: https://seominjoon.github.io
[minjoon-github]: https://github.com/seominjoon
[dev]: https://github.com/allenai/bi-att-flow/tree/dev
