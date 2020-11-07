#!/usr/bin/env python3
# encoding: utf-8
"""
This file defines a sklearn compatible interface for nn_builder.
"""

# ───────────────────────────────── imports ────────────────────────────────── #
# Standard Library
import numpy as np
import tensorflow as tf
from scipy.special import softmax
from sklearn.base import ClassifierMixin, RegressorMixin
import torch.nn as nn
from torchvision import models
import copy
import numpy as np
import torch
import torch.nn as nn
import numpy as np
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
import cv2
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset
# ---------------------------------------------------------------------------- #

from src.utils import args, load_data, logging, load_dataset
from src.utils.load_data import CustomTensorDataset, data2img


class SuperTML(ClassifierMixin):
    def __init__(self,
                 nb_classes: int,
                 base_model: str = 'resnet18',
                 optimiser: str = 'Adagrad',
                 batch_size: int = 16,
                 device: str = 'cpu',
                 epochs: int = 10
                 ):
        self.nb_classes = nb_classes
        self.base_model = base_model
        self.optimiser = optimiser
        self.batch_size = batch_size
        self.device = device
        self.epochs = epochs

        # Model selection
        if self.base_model == 'resnet18':
            self.model = models.resnet18(pretrained=True)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, nb_classes)
        elif self.base_model == 'densenet121':
            self.model = models.densenet121(pretrained=True)
            self.model.classifier = nn.Linear(1024, nb_classes)

        if self.device == 'gpu':
            self.model = nn.DataParallel(self.model.to(device))

    def get_params(self, deep=True):
        return {'nb_classes': self.nb_classes,
                'base_model': self.base_model,
                'optimiser': self.optimiser,
                'batch_size': self.batch_size,
                'device': self.device,
                'epochs': self.epochs}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        self.model = SuperTML(nb_classes=self.nb_classes,
                              base_model=self.base_model,
                              optimiser=self.optimiser,
                              batch_size=self.batch_size,
                              device=self.device,
                              epochs=self.epochs)

        return self

    def __repr__(self, N_CHAR_MAX=700):
        the_class = 'super_tml'
        init_params = self.get_params()
        signature = []
        for k, v in init_params.items():
            signature.append(k + '=' + str(v))
        signature = ', '.join(signature)
        signature = '(' + signature + ')'
        return the_class + signature

    def __hash__(self) -> int:
        return hash(str(self))

    def __getstate__(self):
        state = self.__dict__.copy()
        state['model'] = None
        return state

    def _opt_selection(self):
        if self.optimiser == 'Adamax':
            optimizer = torch.optim.Adamax(self.model.parameters(), lr=0.0001)
        elif self.optimiser == 'Adagrad':
            optimizer = torch.optim.Adagrad(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        elif self.optimiser == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        else:
            raise NotImplementedError
        return optimizer

    def _load_data(self, x_train, y_train=None):
        # Dataset and Dataloader settings
        kwargs = {} if self.device == 'cpu' else {'num_workers': 2, 'pin_memory': True}
        loader_kwargs = {'batch_size': self.batch_size, **kwargs}

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Build Dataset
        train_data = CustomTensorDataset(data=(x_train, y_train), transform=transform)
        # Build Dataloader
        train_loader = DataLoader(train_data, shuffle=True, **loader_kwargs)

        return train_loader

    def fit(self, X, y):
        train_loader = self._load_data(X, y)

        # Optimizer
        optimizer = self._opt_selection()

        # Loss Criterion
        self.criterion = nn.CrossEntropyLoss()

        best_train = 0.0
        epochs = self.epochs + 1
        for epoch in range(1, epochs):
            # Train and Validate
            train_stats = self._train_step(optimizer, train_loader)
            print(train_stats)

            # Keep best model
            if train_stats['accuracy'] >= best_train:
                best_train = train_stats['accuracy']
                self.best_model_weights = copy.deepcopy(self.model.state_dict())

    def _train_step(self, optimizer, train_loader):
        self.model.train()
        avg_loss, avg_acc = 0.0, 0.0
        for i, (x_imgs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            # forward pass
            x_imgs, labels = x_imgs.to(self.device), labels.to(self.device)
            probs = self.model(x_imgs)
            loss = self.criterion(probs, labels)
            # back-prop
            loss.backward()
            optimizer.step()
            # gather statistics
            avg_loss += loss.item()
            _, preds = torch.max(probs, 1)
            avg_acc += torch.sum(preds == labels.data).item()
        return {'loss': avg_loss / len(train_loader), 'accuracy': avg_acc / len(train_loader.dataset)}

    @torch.no_grad()
    def _valid_step(self, val_loader):
        self.model.eval()
        all_preds = []
        for i, (x_imgs) in enumerate(val_loader):
            # forward pass
            x_imgs = x_imgs.to(self.device)
            outputs = self.model(x_imgs)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds)
        all_preds = torch.cat(all_preds, dim=0)
        all_preds = all_preds.cpu().detach().numpy()
        return all_preds

    def predict(self, X):
        # Load best model and evaluate on test set
        self.model.load_state_dict(self.best_model_weights)
        test_loader = self._load_data(X, None)
        return self._valid_step(test_loader)