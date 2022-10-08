# IMPORTS
from email.policy import default
import pickle5 as pickle

# Libraries
import matplotlib.pyplot as plt
import pandas as pd
import wandb
import os
from os.path import exists
import argparse


# Evaluation
from sklearn.metrics import classification_report, precision_recall_curve, auc, roc_auc_score, \
    accuracy_score, f1_score, recall_score, precision_score, confusion_matrix

import wandb
import glob

# data
import numpy as np
import matplotlib.pyplot as plt

#torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from torch.utils.data import WeightedRandomSampler

global activation_fn
activation_fn = nn.Softmax(dim=1)


# Save and Load Functions
def save_checkpoint(save_path, model, optimizer, valid_loss):

    if save_path == None:
        return
    
    state_dict = {'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_loss': valid_loss}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_checkpoint(load_path, model, optimizer, device):

    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    
    return state_dict['valid_loss']


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']



class Dataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels_df = labels[labels.icu_day_id==1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tensor_data = torch.tensor(self.data.values[idx][4:], dtype=torch.double)
        stay_id = self.data.values[idx][0]

        AKI_1_label = (np.sum(self.labels_df.AKI_1.values[idx]) > 0).astype(int)
        AKI_2_label = (np.sum(self.labels_df.AKI_2.values[idx]) > 0).astype(int)
        AKI_3_label = (np.sum(self.labels_df.AKI_3.values[idx]) > 0).astype(int)
        NO_AKI_label = (np.sum(self.labels_df.NO_AKI.values[idx]) > 0).astype(int)
        tensor_labels = torch.tensor([AKI_1_label, AKI_2_label, AKI_3_label, NO_AKI_label], dtype=torch.double)

        return tensor_data, tensor_labels
    
        
class EHR_MODEL(nn.Module):
    def __init__(self,  drop=0.6, hidden_size=128):
        super(EHR_MODEL, self).__init__()

        self.p = drop
        self.input_size = 137
        self.H = hidden_size
        # layers of the network
        self.lstm = nn.LSTM(input_size=self.input_size,
                              hidden_size=self.H,
                              num_layers=1,
                              batch_first=True,
                              bidirectional=True)

        self.fc_1 = nn.Linear(2 * self.H , 512)
        self.fc_2 = nn.Linear(512, 4)
        self.drop = nn.Dropout(p=self.p)

    def forward(self, data):
        # data = data.to(torch.double) 
        out_lstm, _ = self.lstm(data.unsqueeze(1))
        out_lstm = out_lstm.reshape(out_lstm.size(0), out_lstm.size(1)*out_lstm.size(2))

        output = self.fc_1(out_lstm)
        output = self.drop(output)
        output = self.fc_2(output)
        output = torch.squeeze(output, 1)

        return output


###########################################################################################################
######################################## TRAIN ###########################################################
def train(model, 
        optimizer,
        train_loader,
        valid_loader,
        file_path,
        device='cuda',
        num_epochs=1,
        criterion = 'BCELoss',
        pos_weight = torch.tensor([]),
        best_valid_loss = float("Inf"),
        dimension=128,
        epoch_patience=15,
        threshold=None,
        scheduler=None,
        activation_fn=activation_fn):

    # initialize running values
    running_loss = 0.0
    running_acc = 0.0
    valid_running_loss = 0.0
    valid_running_acc = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    train_acc_list = []
    valid_acc_list = []
    global_steps_list = []
    stop_training = 0

    # activation_fn = nn.Sigmoid()
    # activation_fn = nn.Softmax(dim=1)

    if criterion == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss()
        criterion.pos_weight = pos_weight.to(device)
        use_sigmoid=False
    else:
        criterion = nn.BCELoss()
        use_sigmoid = True

    # training loop
    for epoch in range(num_epochs):  

        model.train()
        for tensor_data, tensor_labels  in train_loader:
            # transferring everything to GPU
            tensor_labels = tensor_labels.to(device)
            tensor_data = tensor_data.to(device)

            print(f'Step {global_step+1}/{len(train_loader)}')

            output = model(tensor_data)

            if use_sigmoid:
                loss = criterion(activation_fn(output), tensor_labels)
            else:
                loss = criterion(output, tensor_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()            
            global_step += 1

            wandb.log({'step_train_loss': loss.item(), 'global_step': global_step})
            
        if scheduler is not None:
            scheduler.step()
            print(f'Learning rate is {get_lr(optimizer)}')

        model.eval()
        stacked_labels = torch.tensor([]).to(device)
        stacked_probs = torch.tensor([]).to(device)
        with torch.no_grad():
            # validation loop
            for tensor_data, tensor_labels  in valid_loader:
                tensor_labels = tensor_labels.to(device)
                tensor_data = tensor_data.to(device)

                output = model(tensor_data)

                if use_sigmoid:
                    loss = criterion(activation_fn(output), tensor_labels)
                else:
                    loss = criterion(output, tensor_labels)

                valid_running_loss += loss.item()
                probs = activation_fn(output)
                # stacking labels and predictions
                stacked_labels = torch.cat([stacked_labels, tensor_labels], dim=0)
                stacked_probs = torch.cat([stacked_probs, probs], dim=0, )

        # transfer to device
        stacked_labels = stacked_labels.cpu().detach().numpy()
        stacked_probs = stacked_probs.cpu().detach().numpy()
        # valid loss
        epoch_average_train_loss = running_loss / len(train_loader)  
        epoch_average_valid_loss = valid_running_loss / len(valid_loader)

        train_loss_list.append(epoch_average_train_loss)
        valid_loss_list.append(epoch_average_valid_loss)


        y_true = np.argmax(stacked_labels, axis=1)
        y_pred = np.argmax(stacked_probs, axis=1)

        f1_score_ = np.round(f1_score(y_true, y_pred, average='macro', zero_division=0), 2)
        recall_score_ = np.round(recall_score(y_true, y_pred, average='macro', zero_division=0), 2)
        precision_score_ = np.round(precision_score(y_true, y_pred, average='macro', zero_division=0), 2)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity =  np.round(tn / (tn + fp), 2)
        pr_auc = np.round(auc(recall_score_, precision_score_), 2)
        wandb.log({'val_f1_score': f1_score_, 'val_recall_score':recall_score_, \
                    'val_specificity':specificity, 'val_pr_auc':pr_auc,\
                        'epoch': epoch+1})

        global_steps_list.append(global_step)
        wandb.log({'epoch_average_train_loss': epoch_average_train_loss,
                    'epoch_average_valid_loss': epoch_average_valid_loss,
                    'epoch': epoch+1})

        # resetting running values
        running_loss = 0.0                
        valid_running_loss = 0.0
        
        # print progress
        print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                .format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader),
                        epoch_average_train_loss, epoch_average_valid_loss))      

        # checkpoint
        if best_valid_loss > epoch_average_valid_loss:
            best_valid_loss = epoch_average_valid_loss
            save_checkpoint(file_path + '/model.pt', model, optimizer, best_valid_loss)
            stop_training = 0
        else:
            stop_training +=1
        
        if stop_training == epoch_patience:
            break
    print('Finished Training!')

######################################## EVALUATION ###########################################################

def evaluate(model, test_loader, threshold=None, log_res=True, activation_fn = nn.Softmax(dim=1)):
    print('Evaluation..')
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    device = 'cpu'
    model = model.to(device)
    stacked_labels = torch.tensor([]).to(device)
    stacked_probs = torch.tensor([]).to(device)
    
    model.eval()
    step = 1
    with torch.no_grad():
        for  tensor_data, tensor_labels  in test_loader:
            if step % 100==0:
                print(f'Step {step}/{len(test_loader)}' )

            tensor_data = tensor_data.to(device)
            tensor_labels = tensor_labels.to(device)

            probs = model(tensor_data)
            probs = activation_fn(probs)
            # output = (probs > threshold).int()

            # stacking labels and predictions
            stacked_labels = torch.cat([stacked_labels, tensor_labels], dim=0, )
            # stacked_preds = torch.cat([stacked_preds, output], dim=0, )
            stacked_probs = torch.cat([stacked_probs, probs], dim=0, )
            step += 1
            
    # transfer to device
    stacked_labels = stacked_labels.cpu().detach().numpy()
    stacked_probs = stacked_probs.cpu().detach().numpy()

    y_true = np.argmax(np.argmax(stacked_labels, axis=1), axis=1)
    y_pred = np.argmax(stacked_probs, axis=1)

    accuracy = np.round(accuracy_score(y_true, y_pred), 2)
    print(f'Accuracy: {accuracy}')

    f1_score_ = np.round(f1_score(y_true, y_pred,  average='macro', zero_division=0), 2)
    print(f'F1: ', f1_score_)

    recall_score_ = np.round(recall_score(y_true, y_pred, average='macro', zero_division=0), 2)
    print(f'Sensitivity: ', recall_score_)

    precision_score_ = np.round(precision_score(y_true, y_pred, average='macro', zero_division=0), 2)
    print(f'Precision: ', precision_score_)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity =  np.round(tn / (tn + fp), 2)
    print(f'Specificity: ', specificity)

    pr_auc = np.round(auc(recall_score_, precision_score_), 2) 
    print(f'PR AUC: ', pr_auc)

    roc_auc = np.round(roc_auc_score(y_true, y_pred), 2)
    print(f'ROC AUC: ', roc_auc)
    # confusion matrix
    print(f'Confusion matrix:\n', confusion_matrix(y_true, y_pred))
    # get classification metrics for all samples in the test set
    classification_report_res = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    print(classification_report(y_true, y_pred, zero_division=0, output_dict=False))


    if log_res:
        wandb.log({'test_accuracy' :accuracy, 'test_f1_score':f1_score_, \
                    'test_recall_score':recall_score_, 'test_precision_score':precision_score_, \
                        'test_specificity':specificity})

    return



def main(saving_folder_name=None, additional_name='', criterion='BCELoss', \
    use_gpu=True, project_name='test', experiment='test', oversampling=False,\
            pred_window=2, observing_window=2, BATCH_SIZE=128, LR=0.0001, hidden_size=128,\
                drop=0.6, weight_decay=0, num_epochs=1, wandb_mode='disabled', PRETRAINED_PATH=None, run_id=None):

    # define the device
    if use_gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device='cpu'
    print(f'Device: {device}')         

    CURR_PATH = '/home/svetlanamaslenkova/Documents/AKI_deep/LSTM'
    DF_PATH = CURR_PATH +'/dataframes_2/'
    destination_folder = CURR_PATH + '/icu_training/'
    LABELS_PATH = '/home/svetlanamaslenkova/Documents/AKI_deep/LSTM/pickles_2/aki_stage_labels.pkl'

    # CURR_PATH = os.getcwd()
    # DF_PATH = CURR_PATH +'icu_data/dataframes_2/'
    # destination_folder = '/l/users/svetlana.maslenkova/models' + '/icu_models/no_pretraining/'

    with open(LABELS_PATH, 'rb') as f:
        aki_stage_labels = pickle.load(f)

    with open(DF_PATH + 'train_data_icu_multihot.pkl', 'rb') as f:
        train_data = pickle.load(f)

    with open(DF_PATH + 'val_data_icu_multihot.pkl', 'rb') as f:
        val_data = pickle.load(f)
    
    with open(DF_PATH + 'test_data_icu_multihot.pkl', 'rb') as f:
        test_data = pickle.load(f)


    train_dataset = Dataset(train_data, aki_stage_labels)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = Dataset(test_data, aki_stage_labels)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = Dataset(val_data, aki_stage_labels)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    if oversampling:
        a = 0

    model = model = EHR_MODEL().to(torch.double).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    if PRETRAINED_PATH is not None:
        load_checkpoint(PRETRAINED_PATH, model, optimizer, device)

    exp_lr_scheduler = None

    train_params = {
                'model':model,
                'device':device,
                'optimizer':optimizer,
                'criterion':criterion,
                'train_loader':val_loader,
                'valid_loader':val_loader,
                'num_epochs':num_epochs, 
                'file_path':destination_folder,
                'best_valid_loss':float("Inf"),
                'dimension':128,
                'epoch_patience':15,
                'threshold':None,
                'scheduler':exp_lr_scheduler
            }

    weights = ''
    # path for the model
    if saving_folder_name is None:
        saving_folder_name = additional_name + '_mul_hot_' + 'ICU_' + experiment + '_' + str(len(train_dataset) // 1000) + 'k_'  \
            + 'lr' + str(LR) + '_h'+ str(hidden_size) + '_pw' + str(pred_window) + '_ow' + str(observing_window) \
                + '_wd' + str(weight_decay) + '_'+ weights + '_drop' + str(drop) 
                

    
    file_path = destination_folder + saving_folder_name
    train_params['file_path'] = file_path

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    # wandb setup
    os.environ['WANDB_API_KEY'] = '8e859a0fc58f296096842a367ca532717d3b4059'
    run_name = saving_folder_name
    if run_id is None:    
        run_id = wandb.util.generate_id()  
        resume = 'allow' 
    else:
        resume = 'must'

    args = {'optimizer':'Adam', 'criterion':'BCELoss', 'LR':LR, 'hidden_size':hidden_size, \
            'pred_window':pred_window, 'experiment':experiment,'weight_decay':weight_decay, 'drop':drop}

    wandb.init(project=project_name, name=run_name, mode=wandb_mode, config=args, id=run_id, resume=resume)
    print('Run id is: ', run_id)
    print('Run name: ', run_name)
    print(f'\n\nMODEL PATH: {file_path}')
    # training
    print('Training started..')
    train(**train_params)

    # testing
    print('\nTesting the model...')
    load_checkpoint(file_path + '/model.pt', model, optimizer, device=device)
    evaluate(model, test_loader, threshold=None, log_res=True)

    wandb.finish()


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--additional_name', type=str, default="", help="The options are: 1_,")
    parser.add_argument('--lr', type=float, default=0.00001, help="This is the learning rate.")
    parser.add_argument('--drop', type=float, default=0.6, help="This is the dropout probability.")


    return parser.parse_known_args()

args, _ = _parse_args()

print(args)


main(saving_folder_name=None, additional_name='', criterion='BCELoss', \
    use_gpu=True, project_name='test', experiment='test', oversampling=False,\
            pred_window=2, observing_window=2, BATCH_SIZE=1024, LR=0.0001, hidden_size=128,\
                drop=0.6, weight_decay=0, num_epochs=1, wandb_mode='disabled', PRETRAINED_PATH=None, run_id=None)
