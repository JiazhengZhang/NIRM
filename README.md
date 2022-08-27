# NIRM

The official PyTorch implementation of Neural Influence Ranking Model (NIRM) in the following paper:

```
Jiazheng Zhang, Bang Wang. 2022. Dismantling Complex Networks by a Neural Model 
Trained from Tiny Networks. In CIKM'22, October 17-22, 2022, Atlanta, USA, 10 pages.
```

## Dependencies

- torch 1.7.1
- torch-geometric 1.7.2
- torch-sparse 0.6.9
- torch-scatter 2.0.7
- sklearn 0.24.2
- numpy 1.19.1
- pandas 1.3.0
- networkx 2.6.2
- scipy 1.7.0

Install all dependencies using
```
pip install -r requirements.txt
```


## Usage
1.  Generate synthetic training dataset:

```
python GenerateTrainData.py
```

2.  Modify hyper-parameters in Train.py, and run the following to train the model:

```
python Train.py
```

3.  Test the well-trained model on the real-world networks:

```
python Test.py
```
we provide a well-trained model for one-pass dismantling in the fold './checkpoints/'.



## Citation

Please cite our work if you find our code/paper is helpful to your work.

```
@inproceedings{zhang2022NIRM,
  title={Dismantling Complex Networks by a Neural Model Trained from Tiny Networks},
  author={Zhang, Jiazheng and Wang, Bang},
  booktitle={Proceedings of the 31st ACM International Conference on Information and Knowledge Management},
  series={CIKM'22},
  year={2022},
  location={Atlanta, Georgia, USA},
  numpages={10}
}
```
