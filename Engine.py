import torch
import numpy as np
import copy

from tqdm import tqdm

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate(model, dataloader, device = 'cuda'):

    model = model.to(device)
    model.eval()
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    pbar.set_description("Test")
    sum_loss = 0.0
    T_list = []
    T_logits_pred_list = []
    with torch.no_grad():
        for batch_idx, batch in pbar:
            X, T, y = batch # X: (batch_size, bag_size, 3, 512, 512), T: (batch_size, bag_size), y: (batch_size, 1), L: (batch_size, bag_size, bag_size), mask: (batch_size, bag_size)
            X = X.to(device)
            T = T.to(device)

            T_logits_pred, S_pred = model(X)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(T_logits_pred.float(), T.float())

            sum_loss += loss.item()
            T_list.append(T.cpu().numpy())
            T_logits_pred_list.append(T_logits_pred.cpu().numpy())

    T = np.concatenate(T_list) # (batch_size*bag_size,)

    T_logits_pred = np.concatenate(T_logits_pred_list) # (batch_size,)
    T_pred = np.where(T_logits_pred > 0, 1, 0) # (batch_size, 1)

    metrics = {}
    metrics['loss'] = sum_loss / len(dataloader)
    try: 
        metrics['bag/auc'] = roc_auc_score(T, T_logits_pred)
    except ValueError:
        metrics['bag/auc'] = 0.0
    metrics['bag/acc'] = accuracy_score(T, T_pred)
    metrics['bag/prec'] = precision_score(T, T_pred, zero_division=0)
    metrics['bag/rec'] = recall_score(T, T_pred, zero_division=0)
    metrics['bag/f1'] = f1_score(T, T_pred, zero_division=0)

    metrics = {f'test/{k}' : v for k, v in metrics.items()}

    return metrics

class Trainer:
    def __init__(self, model, criterion, optimizer, device='cpu', early_stop_patience = None, lr_decay = None):
        self.model = model  
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_decay = lr_decay
        if self.lr_decay is not None:
            self.lr_sch = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=self.lr_decay, patience=2, verbose=True)
        else:
            self.lr_sch = None
        self.device = device
        self.early_stop_patience = early_stop_patience

        if self.early_stop_patience is None:
            self.early_stop_patience = np.inf

        self.best_model = None
        self.best_bag_auc = None
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)

    def train(self, max_epochs, train_dataloader, val_dataloader=None):

        if val_dataloader is None:
            val_dataloader = train_dataloader
        
        if self.best_model is None:
            self.best_model = self.copy_model()
        
        if self.best_bag_auc is None:
            self.best_bag_auc = -np.inf

        early_stop_count = 0
        for epoch in range(1, max_epochs+1):

            # Train loop
            self.model.train()
            pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
            pbar.set_description(f"Epoch {epoch} - Train")
            sum_loss = 0.0
            T_list = []
            T_logits_pred_list = []
            for batch_idx, batch in pbar:
                X, T, y = batch # X: (batch_size, bag_size, 3, 512, 512), T: (batch_size, bag_size), y: (batch_size, 1), L: (batch_size, bag_size, bag_size), mask: (batch_size, bag_size)
                X = X.to(self.device)
                T = T.to(self.device)

                self.optimizer.zero_grad()

                T_logits_pred, S_pred = self.model(X)
                loss = self.criterion(T_logits_pred.float(), T.float())
                loss.backward()
                self.optimizer.step()

                sum_loss += loss.item() / X.shape[0]
                T_list.append(T.detach().cpu().numpy())
                T_logits_pred_list.append(T_logits_pred.detach().cpu().numpy())

                if batch_idx < (len(train_dataloader) - 1):
                    pbar.set_postfix({'train/loss' : sum_loss / (batch_idx + 1)})
                else:
                    T = np.concatenate(T_list) # (batch_size*bag_size,)
                    T_logits_pred = np.concatenate(T_logits_pred_list) # (batch_size*bag_size,)
                    
                    try:
                        auc_score = roc_auc_score(T, T_logits_pred)
                    except:
                        auc_score = 0.0

                    train_metrics = {'train/loss' : sum_loss / (batch_idx + 1), 'train/bag/auc' : auc_score}
                    pbar.set_postfix(train_metrics)
            pbar.close()

            # Validation loop
            self.model.eval()
            pbar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
            pbar.set_description(f"Epoch {epoch} - Validation")
            sum_loss = 0.0
            T_list = []
            T_logits_pred_list = []
            with torch.no_grad():
                for batch_idx, batch in pbar:
                    X, T, y = batch # X: (batch_size, bag_size, 3, 512, 512), T: (batch_size, bag_size), y: (batch_size, 1), L: (batch_size, bag_size, bag_size), mask: (batch_size, bag_size)
                    X = X.to(self.device)
                    T = T.to(self.device)

                    T_logits_pred, S_pred = self.model(X)
                    T_logits_pred = T_logits_pred.detach()
                    loss = self.criterion(T_logits_pred.float(), T.float())

                    sum_loss += loss.item() / X.shape[0]
                    T_list.append(T.detach().cpu().numpy())
                    T_logits_pred_list.append(T_logits_pred.detach().cpu().numpy())

                    if batch_idx < (len(val_dataloader) - 1):
                        pbar.set_postfix({'val/loss' : sum_loss / (batch_idx + 1)})
                    else:
                        T = np.concatenate(T_list) # (batch_size, 1)
                        T_logits_pred = np.concatenate(T_logits_pred_list) # (batch_size, 1)
                        
                        try:
                            auc_score = roc_auc_score(T, T_logits_pred)
                        except:
                            auc_score = 0.0                              
                        
                        val_metrics = {'val/loss' : sum_loss / (batch_idx + 1), 'val/bag/auc' : auc_score}
                        pbar.set_postfix(val_metrics)
                
                if self.lr_sch is not None:
                    self.lr_sch.step(val_metrics['val/bag/auc'])

                if val_metrics['val/bag/auc'] < self.best_bag_auc:
                    early_stop_count += 1
                    print(f'Early stopping count: {early_stop_count}')
                else:
                    self.best_bag_auc = val_metrics['val/bag/auc']
                    del self.best_model
                    self.best_model = self.copy_model()
                    early_stop_count = 0
                
            if early_stop_count >= self.early_stop_patience:
                print('Reached early stopping condition')
                break

            pbar.close()
    
    def copy_model(self):
        model_copy = copy.deepcopy(self.model.to('cpu')).to('cpu')
        self.model = self.model.to(self.device)
        return model_copy
    
    def get_best_model(self):
        return self.best_model
