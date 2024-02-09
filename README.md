# GraphDTAF
## About GraphDTAF

DeepDTAF is a deep learning architecture.  

The benchmark dataset can be found in `./data/`. The DeepDTAF model is available in `./src/`. And the result will be generated in `./runs/`. See our paper for more details.

### Requirements:
- python 3.7
- cudatoolkit 10.1.243
- cudnn 7.6.0
- pytorch 1.4.0
- numpy 1.16.4
- scikit-learn 0.21.2
- pandas 0.24.2
- tensorboard 2.0.0
- scipy 1.3.0
- numba 0.44.1
- tqdm 4.32.1


### Training & Evaluation

to train your own model
```bash
cd ./src/
python main3.py
```
to see the result
```bash
tensorboard ../runs/DeepDTAF_<datetime>_<seed>/
```
