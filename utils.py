import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import torch.optim as optim
from kan import *
# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Function to compute all regression metrics
def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    return {
        'ACC': acc,
        'Prec': prec,
        'Rec': rec,
        'F1' : f1,
        'AUC': auc
    }


# Train a classification model
def train_calssification(model, num_epochs, train_dl, x_test,y_test,loss_fn,optimizer):
    loss_hist_train = [0] * num_epochs
    accuracy_hist_train = [0] * num_epochs
    loss_hist_valid = [0] * num_epochs
    accuracy_hist_valid = [0] * num_epochs
    for epoch in range(num_epochs):
        model.train()
        for x_batch,y_batch in train_dl:        
            pred = model(x_batch.to(device))
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_hist_train[epoch] += (loss.item()*y_batch.size(0))
            is_correct = ((pred>=0.5).float().detach().cpu() == y_batch.detach().cpu()).float()
            accuracy_hist_train[epoch] += is_correct.sum()
        loss_hist_train[epoch] /= len(train_dl.dataset)
        accuracy_hist_train[epoch] /= len(train_dl.dataset)
        model.eval()
        with torch.no_grad():
            pred = model(x_test.to(device))
            loss = loss_fn(pred, y_test.to(device))
            loss_hist_valid[epoch] += loss.item()
            is_correct = ((pred>=0.5).float().detach().cpu() == y_test).float()
            accuracy_hist_valid[epoch] += is_correct.mean()
        if epoch%10==0: 
            print(f'Epoch {epoch+1} accuracy: 'f'{accuracy_hist_train[epoch]:.4f}, val_accuracy: 'f'{accuracy_hist_valid[epoch]:.4f}, loss: 'f'{loss_hist_train[epoch]:.4f}, val_loss: 'f'{loss_hist_valid[epoch]:.4f}')
    
    fig = plt.figure(figsize=(16, 4))
    ax = fig.add_subplot(1, 2, 1)
    plt.plot(loss_hist_train, lw=4)
    plt.plot(loss_hist_valid, lw=4)
    plt.legend(['Train loss', 'Validation loss'], fontsize=15)
    ax.set_xlabel('Epochs', size=15)
    ax = fig.add_subplot(1, 2, 2)
    plt.plot(accuracy_hist_train, lw=4)
    plt.plot(accuracy_hist_valid, lw=4)
    plt.legend(['Train acc.', 'Validation acc.'], fontsize=15)
    ax.set_xlabel('Epochs', size=15)
    
    return loss_hist_train, loss_hist_valid,accuracy_hist_train, accuracy_hist_valid
