import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def plot_bag_MNIST(bag, obj_labels=None, att_values=None, figsize=None):
    fig, ax = plt.subplots(1, len(bag), figsize=figsize)
    for i, img in enumerate(bag):
        ax[i].imshow(img[0], cmap='gray')
        title = ''
        if obj_labels is not None:
            title = f'label: {obj_labels[i].item()}'
        if att_values is not None:
            title += f'\natt: {att_values[i].item():.3f}'
        ax[i].set_xlabel(title)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    plt.savefig(f"../../Test/MNIST/MNIST-bag-{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}.jpg", bbox_inches='tight')
    plt.show()


def plot_bag_RSNA(bag, bag_label=None, obj_labels=None, att_values=None, figsize=None):
    division = 6
    fig, ax = plt.subplots(int(len(bag)/division), division, figsize=figsize)
    for j in range(division):
        for i in range(int(len(bag)/division)):
            ax[i,j].imshow(bag[division*i+j][0], cmap='inferno', aspect="auto")
            title = ''
            if obj_labels is not None:
                title = f'label: {obj_labels[division*i+j]}'
            if att_values is not None:
                title += f' / att: {att_values[division*i+j]:.3f}'
            ax[i,j].set_xlabel(title, fontsize=5)
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])
    plt.suptitle(f'Bag label: {bag_label}', fontsize=15, y=0.92)
    plt.savefig(f"../../Test/RSNA/RSNA-bag-{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}.jpg", bbox_inches='tight')
    plt.show()


def plot_att_MNIST(S_pred, y_label, thr=None):
    fig, ax = plt.subplots()
    max = np.max([np.max(S_pred)])
    ax.plot(S_pred, 'o-', label='Attention')
    if thr is not None:
        ax.plot([thr] * len(S_pred), 'r--', label='Threshold')
    ax.fill_between(np.arange(len(S_pred)), 0, S_pred, alpha=0.2)
    ax.fill_between(np.arange(len(S_pred)), 0, y_label, alpha=0.2, step='pre', label='Positive instances')
    ax.set_xlabel('Instance index')
    ax.set_ylabel('Attention score')
    ax.set_title('Attention scores')
    ax.set_ylim(0, 1.01*max)
    ax.legend()
    fig.savefig(f"../../Test/MNIST/MNIST-att-{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}.jpg", bbox_inches='tight')
    return fig


def plot_att_RSNA(S_pred, bag_label, y_label, thr=None):
    fig, ax = plt.subplots()
    max = np.max([np.max(S_pred)])
    ax.plot(S_pred, 'o-', label='Attention')
    if thr is not None:
        ax.plot([thr] * len(S_pred), 'r--', label='Threshold')
    ax.fill_between(np.arange(len(S_pred)), 0, S_pred, alpha=0.2)
    ax.fill_between(np.arange(len(S_pred)), 0, y_label, alpha=0.2, step='pre', label='Positive instances')
    ax.set_xlabel('Instance index')
    ax.set_ylabel('Attention score')
    ax.set_title(f'Attention scores - Bag label: {bag_label}')
    ax.set_ylim(0, 1.01*max)
    ax.legend()
    fig.savefig(f"../../Test/RSNA/RSNA-att-{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}.jpg", bbox_inches='tight')
    return fig


def plot_att_dist(model, dataset, device, dataset_name):

    model = model.to(device)
    model.eval()
    pos_inst_list = []
    neg_inst_list = []
    for i in range(len(dataset)):
        data, T_label, y_labels = dataset[i]
        bag_size = len(y_labels)
        T_logits, S_pred = model(data.unsqueeze(dim=0).to(device))
        y_labels = y_labels.detach().cpu().numpy().astype(int).squeeze()
        S = S_pred.detach().cpu().numpy().astype(float).squeeze()
        S = (S - S.min()) / (S.max() - S.min())
        if T_label == 1:
            for j in range(len(y_labels)):
                if y_labels[j] == 1:
                    pos_inst_list.append([S[j]])
                else:
                    neg_inst_list.append([S[j]])
    
    fig, ax = plt.subplots()
    counts, bins = np.histogram(neg_inst_list, bins=20)
    ax.hist(bins[:-1], bins, weights=counts, label='Negative instances', edgecolor='black', alpha=0.8)
    counts, bins = np.histogram(pos_inst_list, bins=20)
    ax.hist(bins[:-1], bins, weights=counts, label='Positive instances', edgecolor='black', alpha=0.8)
    ax.set_xlabel('Attention score')
    ax.set_ylabel('Frequency')
    ax.legend()

    fig.savefig(f"../../Test/{dataset_name.upper()}/{dataset_name}-att-dist-{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}.jpg", bbox_inches='tight')
    return fig