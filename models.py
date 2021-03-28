import torch
from torch import nn
import time
import numpy as np
from progress.bar import IncrementalBar
import xgboost as xgb
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import pickle
from sklearn.ensemble import RandomForestClassifier


class FCModel:
    def __init__(self):
        self.model = nn.Sequential(
            nn.Linear(63, 40),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(40, 27),
            nn.ReLU(),
            nn.Softmax())
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_func = nn.CrossEntropyLoss()

    def save(self, checkpoint_path):
        # state_dict: a Python dictionary object that:
        # - for a model, maps each layer to its parameter tensor;
        # - for an optimizer, contains info about the optimizer’s states and hyperparameters used.
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.opt.state_dict()}
        torch.save(state, checkpoint_path)
        print('model saved to %s' % checkpoint_path)

    def load(self, checkpoint_path):
        state = torch.load(checkpoint_path)
        self.model.load_state_dict(state['state_dict'])
        self.opt.load_state_dict(state['optimizer'])
        print('model loaded from %s' % checkpoint_path)

    def forward(self, tensor):
        out = self.model(tensor.to(self.device))
        return out.max(dim=1)[1]

    def train(self, train_loader, val_loader, n_epochs: int):
        '''
        model: нейросеть для обучения,
        train_loader, val_loader: загрузчики данных
        loss_fn: целевая метрика (которую будем оптимизировать)
        opt: оптимизатор (обновляет веса нейросети)
        n_epochs: кол-во эпох, полных проходов датасета
        '''
        opt = self.opt
        loss_fn = self.loss_func
        device = self.device
        model = self.model

        train_loss = []
        val_loss = []
        val_accuracy = []

        for epoch in range(n_epochs):
            ep_train_loss = []
            ep_val_loss = []
            ep_val_accuracy = []
            start_time = time.time()

            model.train(True)  # enable dropout / batch_norm training behavior
            for X_batch, y_batch in train_loader:
                opt.zero_grad()
                # move data to target device
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                # train on batch: compute loss, calc grads, perform optimizer step and zero the grads
                out = model(X_batch)
                loss = loss_fn(out, y_batch)
                loss.backward()
                opt.step()
                ep_train_loss.append(loss.item())

            model.train(False)  # disable dropout / use averages for batch_norm
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    # move data to target device
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    # train on batch: compute loss, calc grads, perform optimizer step and zero the grads
                    out = model(X_batch)
                    loss = loss_fn(out, y_batch)

                    # compute predictions
                    ep_val_loss.append(loss.item())
                    y_pred = out.max(dim=1)[1]
                    ep_val_accuracy.append(
                        np.sum(y_batch.cpu().numpy() == y_pred.cpu().numpy().astype(float)) / len(y_batch.cpu()))
                    # print the results for this epoch:
            print(f'Epoch {epoch + 1} of {n_epochs} took {time.time() - start_time:.3f}s')

            train_loss.append(np.mean(ep_train_loss))
            val_loss.append(np.mean(ep_val_loss))
            val_accuracy.append(np.mean(ep_val_accuracy))

            print(f"\t  training loss: {train_loss[-1]:.6f}")
            print(f"\tvalidation loss: {val_loss[-1]:.6f}")
            print(f"\tvalidation accuracy: {val_accuracy[-1]:.3f}")

        return train_loss, val_loss, val_accuracy


class XGBModel:
    def __init__(self, model=xgb.XGBClassifier(n_estimators=64, max_depth=16, learning_rate=0.1, subsample=0.5)):
        self.model = model

    def apply_transformation(self, dataset):
        data = list()
        labels = list()
        bar = IncrementalBar("Transforming dataset", max=len(dataset))
        for i in range(len(dataset)):
            el = dataset[i]
            data.append(el[0])
            labels.append(el[1])
            bar.next()
        bar.finish()
        return data, labels

    def fit(self, X_data, y_data):
        self.model.fit(X_data, y_data)

    def predict(self, X_data):
        return self.model.predict(X_data)

    def save(self, checkpoint_path):
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(self.model, f)
        print('model saved to %s' % checkpoint_path)

    def load(self, checkpoint_path):
        with open(checkpoint_path, 'rb') as file:
            self.model = pickle.load(file)
        print('model loaded from %s' % checkpoint_path)


class KNN(XGBModel):
    def __init__(self, model=KNeighborsClassifier(n_neighbors=10)):
        super().__init__(model)


class RandomForestModel:
    def __init__(self, model=RandomForestClassifier(n_estimators=64)):
        self.model = model

    def apply_transformation(self, dataset):
        data = list()
        labels = list()
        bar = IncrementalBar("Transforming dataset", max=len(dataset))
        for i in range(len(dataset)):
            el = dataset[i]
            data.append(el[0])
            labels.append(el[1])
            bar.next()
        bar.finish()
        return data, labels

    def fit(self, X_data, y_data):
        self.model.fit(X_data, y_data)

    def predict(self, X_data):
        return self.model.predict(X_data)

    def save(self, checkpoint_path):
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(self.model, f)
        print('model saved to %s' % checkpoint_path)

    def load(self, checkpoint_path):
        with open(checkpoint_path, 'rb') as file:
            self.model = pickle.load(file)
        print('model loaded from %s' % checkpoint_path)
