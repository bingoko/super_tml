"""
PyTorch implimentation of SuperTML.

Acknowledgement - References:
    •   Baohua Sun. 
        "SuperTML: Two-Dimensional Word Embedding and Transfer Learning Using 
        ImageNet Pretrained CNN Models for the Classifications on Tabular Data".
        CVPR Workshop Paper, 2019

-------------------------
B R A I N C R E A T O R S
-------------------------

Reposetory Author:
    Ioannis Gatopoulos, 2020
"""


from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from src.utils import *
from src.opt.training_pipeline import train_model
from sklearn.model_selection import train_test_split
from src.super_tml import SuperTML

def main():
    fix_random_seed(seed=args.seed)

    # Eneble TensorBoard logs
    writer = SummaryWriter(log_dir='./logs/' +
                           args.dataset + '_' + args.tags +
                           datetime.now().strftime("/%d-%m-%Y/%H-%M-%S"))
    writer.add_text('args', namespace2markdown(args))

    data = pd.read_csv(args.dataset)
    datay = data.loc[:, args.label]
    datax = data.drop(args.label, axis=1)

    datay = datay.to_numpy()
    datax = datax.to_numpy()

    # Split dataset -- Cross Vaidation
    x_train, x_test, y_train, y_test \
        = train_test_split(datax, datay, test_size=0.3, random_state=1)
    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    model = SuperTML(nb_classes=nb_classes,
                 base_model = 'resnet18',
                 optimiser = 'Adagrad',
                 batch_size = 16,
                 device='cpu')
    model.fit(X=x_train, y=y_train)

    writer.close()
    print('\n'+24*'='+' Experiment Ended '+24*'='+'\n')


if __name__ == "__main__":
    main()
