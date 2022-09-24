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

# Tokenization
from tokenizers import  Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Punctuation, Whitespace
from tokenizers.normalizers import Lowercase
from tokenizers import pre_tokenizers, normalizers
from tokenizers.processors import BertProcessing
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


######################################## DATASET ###########################################################
class MyDataset(Dataset):

    def __init__(self, data_path,  tokenizer, labels_df=None, max_length=300, names=True, pred_window=2, observing_window=2):
        # pred_window: number of 12h windows to predict AKI onset
        # observing_window: number of 12h windows to observe
        self.data_path = data_path
        file_list = glob.glob(self.data_path + '*')
        self.data = []
        for sample in file_list:
            self.data.append(sample)

        self.tokenizer = tokenizer
        self.observing_window = observing_window
        self.pred_window = pred_window
        self.max_length = max_length
        self.labels_df = labels_df[labels_df.icu_day_id==1]

        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        return self.make_matrices(idx)
    
    def tokenize(self, text, max_length):
        
        try:
            output = self.tokenizer.encode(text)
        except:
            print( type(text), text, max_length, self.stay_id)
            output = self.tokenizer.encode(text[0])

        # padding and truncation
        if len(output.ids) < max_length:
            len_missing_token = max_length - len(output.ids)
            padding_vec = [self.tokenizer.token_to_id('PAD') for _ in range(len_missing_token)]
            token_output = [*output.ids, *padding_vec]
        elif len(output.ids) > max_length:
            token_output = output.ids[:max_length]
        else:
            token_output = output.ids
        
        return token_output

    def preproccess(self, df):
        df['demographics'] = df['demographics'].fillna('')
        df['previous_diags_codes'] = df['previous_diags_codes'].fillna('')
        df['previous_diags_names'] = df['previous_diags_names'].fillna('')
        df['vitals_names'] = df['vitals_names'].fillna('')
        df['vitals_codes'] = df['vitals_codes'].fillna('')
        df['labs_names'] = df['labs_names'].fillna('')
        df['labs_codes'] = df['labs_codes'].fillna('')
        df['outputs_names'] = df['outputs_names'].fillna('')
        df['outputs_codes'] = df['outputs_codes'].fillna('')
        df['medications_names'] = df['medications_names'].fillna('')
        df['medications_codes'] = df['medications_codes'].fillna('')
        df = df.replace(r';', '', regex=True)
        df['AKI_1'] = df['AKI_1'].fillna(0)
        df['AKI_2'] = df['AKI_2'].fillna(0)
        df['AKI_3'] = df['AKI_3'].fillna(0)

        df = df[(df.icu_12h_window_id.isin(np.arange(self.min_wind, self.min_wind + self.observing_window +  self.pred_window))) | (df.icu_day_id.isin(np.arange(self.min_day, self.observing_window//2 +  self.pred_window//2)))]
        return df

    def make_matrices(self, idx):
        # load csv file
        sample_path = self.data[idx]
        df = pd.read_csv(sample_path)
        # print('Loaded from ', sample_path)
        
        windows_12h = df.icu_12h_window_id.values
        days = df.icu_day_id.values
        self.min_wind = int( np.max([np.min(windows_12h[~np.isnan(windows_12h)]),0] ) )       
        self.min_day = int( np.max( [np.min(days[~np.isnan(days)]), 0] ))  
        self.df = self.preproccess(df)
        self.stay_id = self.df.stay_id.values[0]  
        # print(stay_id)
        sort = np.argsort(self.df.icu_12h_window_id.values)
        windows_12h = self.df.icu_12h_window_id.values[sort]
        days = self.df.icu_day_id.values[sort]

        info_12h = self.df.icu_12h_info_codes.values[sort]
        info_24h_labs = self.df.labs_codes.values[sort]
        info_demo = self.df.demographics.values[0]
        info_diagnoses = self.df.previous_diags_codes.values[0]

        AKI_1_status = self.df.AKI_1.values[sort]
        AKI_2_status = self.df.AKI_2.values[sort]
        AKI_3_status = self.df.AKI_3.values[sort]

        AKI_1_labels_l = []
        AKI_2_labels_l = []
        AKI_3_labels_l = []
        info_12h_list = []
        info_24h_list = []
        day_l = []
        info_l = []
        used_day_id_l = []
        used_wind_id_l = []

        wind_12h_pairs = [(i, i+1) for i in range(0, 2*(self.min_day + self.observing_window//2 +  self.pred_window//2), 2)]

        for day in range(self.min_day, self.min_day + self.observing_window//2 +  self.pred_window//2):
            for wind in wind_12h_pairs[day]:
                if wind not in windows_12h:
                    if day not in days:
                    # print('not in days')
                        AKI_1_labels_l.append(0)
                        AKI_2_labels_l.append(0)
                        AKI_3_labels_l.append(0)
                        day_l.append('')
                        if day not in used_day_id_l:
                            day_l.append('')
                            used_day_id_l.append(day)
                    else:
                        AKI_1_labels_l.append(self.df[self.df.icu_day_id==day].AKI_1.values[0])
                        AKI_2_labels_l.append(self.df[self.df.icu_day_id==day].AKI_2.values[0])
                        AKI_3_labels_l.append(self.df[self.df.icu_day_id==day].AKI_3.values[0])
                        day_l.append(self.df[self.df.icu_day_id==day].icu_12h_info_codes.values[0])
                        if day not in used_day_id_l:
                            day_l.append(self.df[self.df.icu_day_id==day].labs_codes.values[0])
                            used_day_id_l.append(day)
                else:
                    i = list(windows_12h).index(wind)

                    AKI_1_labels_l.append(AKI_1_status[i])
                    AKI_2_labels_l.append(AKI_2_status[i])
                    AKI_3_labels_l.append(AKI_3_status[i])
                    day_l.append(info_12h[i])
                    if day not in used_day_id_l:
                        day_l.append(info_24h_labs[i])
                        used_day_id_l.append(day)
                used_wind_id_l.append(wind)
            day_info = ' '.join(day_l)
            info_l.append(self.tokenize(day_info, self.max_length))

        demographics = info_demo
        diagnoses = info_diagnoses
        demo_diags = ' '.join([demographics, diagnoses])
        demo_diags = self.tokenize(demo_diags, 50)

        #make tensors
        tensor_day_info = torch.tensor(info_l[:self.observing_window//2], dtype=torch.int64)
        tensor_demo_diags = torch.tensor(demo_diags, dtype=torch.int64)

        if self.labels_df is None:
            # making 24h labels from 12h
            AKI_1_labels = [int(bool(np.sum(AKI_1_labels_l[i:i+2]))) for i in np.arange(0, self.observing_window + self.pred_window, 2)]
            AKI_2_labels = [int(bool(np.sum(AKI_2_labels_l[i:i+2]))) for i in np.arange(0, self.observing_window + self.pred_window, 2)]
            AKI_3_labels = [int(bool(np.sum(AKI_3_labels_l[i:i+2]))) for i in np.arange(0, self.observing_window + self.pred_window, 2)]
            
            tensor_labels = torch.tensor([*AKI_1_labels[self.observing_window//2:self.observing_window//2 + self.pred_window//2],\
                                        *AKI_2_labels[self.observing_window//2:self.observing_window//2 + self.pred_window//2],\
                                        *AKI_3_labels[self.observing_window//2:self.observing_window//2 + self.pred_window//2] ]\
                                            , dtype=torch.float64)
        else:
            AKI_1_label = (np.sum(self.labels_df[self.labels_df.stay_id==self.stay_id].AKI_1.values) > 0).astype(int)
            AKI_2_label = (np.sum(self.labels_df[self.labels_df.stay_id==self.stay_id].AKI_2.values) > 0).astype(int)
            AKI_3_label = (np.sum(self.labels_df[self.labels_df.stay_id==self.stay_id].AKI_3.values) > 0).astype(int)
            NO_AKI_label = (np.sum([AKI_1_label, AKI_2_label, AKI_3_label]) == 0).astype(int)
            tensor_labels = torch.tensor([AKI_1_label, AKI_2_label, AKI_3_label, NO_AKI_label])

        return tensor_day_info, tensor_demo_diags, tensor_labels, {'stay_id':self.stay_id, 'day_id':used_day_id_l, 'wind_id':used_wind_id_l}

###################################################### MODEL ###########################################################

class EHR_MODEL(nn.Module):
    def __init__(self, max_length, vocab_size, pred_window=2, observing_window=2,  H=128, embedding_size=200, drop=0.6):
        super(EHR_MODEL, self).__init__()

        self.observing_window = observing_window
        self.pred_window = pred_window
        self.H = H
        self.p = drop
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.max_length_demo_diags = 50
        self.max_length = max_length

        # layers of the network
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.lstm = nn.LSTM(input_size=self.embedding_size,
                              hidden_size=self.H,
                              num_layers=1,
                              batch_first=True,
                              bidirectional=True)

        self.fc_1 = nn.Linear(2 * self.H * (self.max_length_demo_diags + self.max_length) , 2048)
        self.fc_2 = nn.Linear(2048, 4)
        self.drop = nn.Dropout(p=self.p)

    def forward(self, tensor_day_info, tensor_demo_diags):

        out_emb_demo_diags = self.embedding(tensor_demo_diags) # batch_size x max_length_demographics x embedding_size
        out_emb_day_info = torch.squeeze(self.embedding(tensor_day_info), 1) # batch_size x max_length_previous_diags x embedding_size
        embedded = torch.cat([out_emb_demo_diags, out_emb_day_info], dim=1)

        out_lstm, _ = self.lstm(embedded)
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
        scheduler=None):

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
    activation_fn = nn.Softmax(dim=1)

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
        for tensor_day_info, tensor_demo_diags, tensor_labels, _  in train_loader:
            # transferring everything to GPU
            tensor_labels = tensor_labels.to(device)
            tensor_day_info = tensor_day_info.to(device)
            tensor_demo_diags = tensor_demo_diags.to(device)
            print(f'Step {global_step}/{len(train_loader)}')

            output = model(tensor_day_info, tensor_demo_diags)

            if use_sigmoid:
                loss = criterion(activation_fn(output), tensor_labels.type(torch.float32))
            else:
                loss = criterion(output, tensor_labels.type(torch.float32))

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
            for tensor_day_info, tensor_demo_diags, tensor_labels,  _  in valid_loader:
                tensor_labels = tensor_labels.to(device)
                tensor_day_info = tensor_day_info.to(device)
                tensor_demo_diags = tensor_demo_diags.to(device)

                output = model(tensor_day_info, tensor_demo_diags)

                if use_sigmoid:
                    loss = criterion(activation_fn(output), tensor_labels.type(torch.float32))
                else:
                    loss = criterion(output, tensor_labels.type(torch.float32))

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

        stages = ['AKI_1', 'AKI_2', 'AKI_3', 'NO_AKI']
        for w in range(len(stages)):
            stage = stages[w]

            labels = stacked_labels.T[w]
            probs = stacked_probs.T[w]    

            precision, recall, thresholds = precision_recall_curve(labels, probs)
            precision, recall, thresholds = np.round(precision, 2), np.round(recall,2), np.round(thresholds,2)
            
            # convert to f score
            fscore = np.round((2 * precision * recall) / (precision + recall + 0.000001), 2)
            # locate the index of the largest f score
            ix = np.argmax(np.nan_to_num(fscore))
            threshold = np.round(thresholds[ix], 2)

            y_true = labels
            y_pred = (probs > threshold).astype(int)

            f1_score_ = np.round(f1_score(y_true, y_pred, pos_label=1, average='binary', zero_division=0), 2)
            recall_score_ = np.round(recall_score(y_true, y_pred, pos_label=1, average='binary', zero_division=0), 2)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            specificity =  np.round(tn / (tn + fp), 2)
            pr_auc = np.round(auc(recall, precision), 2)
            wandb.log({'val_f1_score_'+stage: f1_score_, 'val_recall_score_'+stage:recall_score_, \
                        'val_specificity'+stage:specificity, 'val_pr_auc'+stage:pr_auc,\
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


# save_metrics(file_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
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
        for  tensor_day_info, tensor_demo_diags,  tensor_labels, _  in test_loader:
            if step % 100==0:
                print(f'Step {step}/{len(test_loader)}' )

            tensor_day_info = tensor_day_info.to(device)
            tensor_demo_diags = tensor_demo_diags.to(device)
            tensor_labels = tensor_labels.to(device)

            probs = model(tensor_day_info, tensor_demo_diags)
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
    # for printing purposes
    stages_names = ['1', '2', '3', 'ANY']
    if threshold==None:
        for w in range(len(stages_names)):
            print('------------- AKI stage ', stages_names[w], '------------- ')

            labels = stacked_labels.T[w]
            probs = stacked_probs.T[w]            

            precision, recall, thresholds = precision_recall_curve(labels, probs)
            precision, recall, thresholds = np.round(precision, 2), np.round(recall,2), np.round(thresholds,2)
            
            # convert to f score
            fscore = np.round((2 * precision * recall) / (precision + recall + 0.000001), 2)

            # locate the index of the largest f score
            ix = np.argmax(np.nan_to_num(fscore))
            threshold = np.round(thresholds[ix], 2)
            print('Best Threshold=%.2f, F-Score=%.2f' % (threshold, fscore[ix]))

            stacked_preds = (probs > threshold).astype(int)
            y_true = labels
            y_pred = stacked_preds

            accuracy = np.round(accuracy_score(y_true, y_pred), 2)
            print(f'Accuracy: {accuracy}')

            f1_score_ = np.round(f1_score(y_true, y_pred, pos_label=1, average='binary', zero_division=0), 2)
            print(f'F1: ', f1_score_)

            recall_score_ = np.round(recall_score(y_true, y_pred, pos_label=1, average='binary', zero_division=0), 2)
            print(f'Sensitivity: ', recall_score_)

            precision_score_ = np.round(precision_score(y_true, y_pred, pos_label=1, average='binary', zero_division=0), 2)
            print(f'Precision: ', precision_score_)

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            specificity =  np.round(tn / (tn + fp), 2)
            print(f'Specificity: ', specificity)

            pr_auc = np.round(auc(recall, precision), 2) 
            print(f'PR AUC: ', pr_auc)

            roc_auc = np.round(roc_auc_score(y_true, y_pred), 2)
            print(f'ROC AUC: ', roc_auc)
            # confusion matrix
            print(f'Confusion matrix:\n', confusion_matrix(y_true, y_pred))
            # get classification metrics for all samples in the test set
            classification_report_res = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
            print(classification_report(y_true, y_pred, zero_division=0, output_dict=False))
            # operating points 
            precision_list = [0.2, 0.25, 0.33, 0.4, 0.5, 0.6, 0.75]
            for p in precision_list:
                idx = find_nearest(precision, p)
                sensitivity = recall[idx]
                print(f'Precision {np.round(precision[idx]*100, 1)}% , Sensitivity {sensitivity} ')

            if log_res:
                wandb.log({'test_accuracy'+stages_names[w] :accuracy, 'test_f1_score'+stages_names[w]:f1_score_, \
                            'test_recall_score'+stages_names[w]:recall_score_, 'test_precision_score'+stages_names[w]:precision_score_, \
                                'test_specificity'+stages_names[w]:specificity})

    return

######################################## MAIN ###########################################################
def main(saving_folder_name=None, additional_name='', criterion='BCELoss', \
    use_gpu=True, project_name='test', experiment='test', oversampling=False,\
            pred_window=2, observing_window=2, BATCH_SIZE=128, LR=0.0001, min_frequency=1, hidden_size=128,\
                drop=0.6, weight_decay=0, num_epochs=1, embedding_size=200, dimension = 128, wandb_mode='disabled', PRETRAINED_PATH=None, run_id=None):

    # define the device
    if use_gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device='cpu'
    print(f'Device: {device}')         

    CURR_PATH = '/home/svetlanamaslenkova/Documents/AKI_deep/LSTM'
    DF_PATH = CURR_PATH +'/dataframes_2/'
    destination_folder = CURR_PATH + '/icu_training/'
    TXT_FILES_CODES_PATH = '/home/svetlanamaslenkova/Documents/AKI_deep/LSTM/txt_files/icu_train'
    TOKENIZER_CODES_PATH = '/home/svetlanamaslenkova/Documents/AKI_deep/LSTM/aki_prediction/tokenizer_icu_codes.json'
    test_data_path ='/home/svetlanamaslenkova/Documents/AKI_deep/LSTM/dataframes_2/test_data/'
    train_data_path ='/home/svetlanamaslenkova/Documents/AKI_deep/LSTM/dataframes_2/train_data/'
    val_data_path ='/home/svetlanamaslenkova/Documents/AKI_deep/LSTM/dataframes_2/val_data/'
    LABELS_PATH = '/home/svetlanamaslenkova/Documents/AKI_deep/LSTM/pickles_2/aki_stage_labels.pkl'

    with open(LABELS_PATH, 'rb') as f:
        aki_stage_labels = pickle.load(f)

    # CURR_PATH = os.getcwd()
    # DF_PATH = CURR_PATH +'icu_data/dataframes_2/'
    # destination_folder = '/l/users/svetlana.maslenkova/models' + '/icu_models/no_pretraining/'
    # TXT_FILES_CODES_PATH = CURR_PATH + '/aki_prediction/txt_files/icu_train'
    # TOKENIZER_CODES_PATH = CURR_PATH + '/aki_prediction/tokenizer_icu_codes.json'
    # test_data_path = CURR_PATH + '/icu_data/dataframes_2/test_data/'
    # train_data_path = CURR_PATH + '/icu_data/dataframes_2/train_data/'
    # val_data_path = CURR_PATH + '/icu_data/dataframes_2/val_data/'
    

    # Training the tokenizer
    print(f'Current directory: {CURR_PATH}')
    if exists(TOKENIZER_CODES_PATH):
        tokenizer = Tokenizer.from_file(TOKENIZER_CODES_PATH)
        print(f'Tokenizer is loaded from ==> {TOKENIZER_CODES_PATH}/tokenizer.json. Vocab size is {tokenizer.get_vocab_size()}')

    vocab_size = tokenizer.get_vocab_size()
    max_length = 350

    train_dataset = MyDataset(data_path=train_data_path, labels_df=aki_stage_labels, tokenizer=tokenizer, max_length=max_length)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = MyDataset(data_path=test_data_path, labels_df=aki_stage_labels, tokenizer=tokenizer, max_length=max_length)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = MyDataset(data_path=val_data_path, labels_df=aki_stage_labels, tokenizer=tokenizer, max_length=max_length)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    if oversampling:
        i = 0
        for _, _, _, _, tensor_labels, _  in train_loader:
            if i == 0:
                stacked_labels = tensor_labels
            else:
                stacked_labels = np.concatenate([stacked_labels, tensor_labels])
            i += 1
        y_train = stacked_labels.T[0]
        count=Counter(y_train)
        class_count=np.array([count[0], count[1]])
        weight=1./class_count
        print('weights for oversampling: ', weight)

        samples_weight = np.array([weight[int(t)] for t in y_train])
        samples_weight=torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, num_workers=1, sampler=sampler)

    model = EHR_MODEL(max_length=max_length, vocab_size=vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    if PRETRAINED_PATH is not None:
        load_checkpoint(PRETRAINED_PATH, model, optimizer, device)

    exp_lr_scheduler = None

    train_params = {
                'model':model,
                'device':device,
                'optimizer':optimizer,
                'criterion':criterion,
                'train_loader':train_loader,
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
        saving_folder_name = additional_name + '_M2_' + 'ICU_' + experiment + '_' + str(len(train_dataset) // 1000) + 'k_'  \
            + 'lr' + str(LR) + '_h'+ str(hidden_size) + '_pw' + str(pred_window) + '_ow' + str(observing_window) \
                + '_wd' + str(weight_decay) + '_'+ weights + '_drop' + str(drop) + '_emb' + str(embedding_size) + '_vocab' + str(vocab_size)
                

    
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

    args = {'optimizer':'Adam', 'criterion':'BCELoss', 'LR':LR, 'min_frequency':min_frequency, 'hidden_size':hidden_size, \
            'pred_window':pred_window, 'experiment':experiment,'weight_decay':weight_decay, 'drop':drop, 'embedding_size':embedding_size, \
                'vocab_size':vocab_size}

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
    parser.add_argument('--embedding_size', type=int, default=200, help="This is the embedding size.")

    return parser.parse_known_args()

args, _ = _parse_args()

print(args)

# test run
main(saving_folder_name=None, additional_name='', criterion='BCELoss', \
    use_gpu=False, project_name='test', experiment='test', oversampling=False, 
            pred_window=2, observing_window=2, BATCH_SIZE=1500, LR=0.0001, min_frequency=5, hidden_size=128,\
                drop=0.6, weight_decay=0, num_epochs=1, embedding_size=200,\
                     wandb_mode='disabled', PRETRAINED_PATH=None, run_id=None)



# #  60277, 60278, 60280
# main(saving_folder_name=None, additional_name=args.additional_name, criterion='BCELoss', \
#     use_gpu=True, project_name='ICU_lstm_model', experiment='no_pretraining', oversampling=False, \
#             pred_window=2, observing_window=2, BATCH_SIZE=1600, LR=args.lr, min_frequency=5, hidden_size=128,\
#                 drop=0.4, weight_decay=0, num_epochs=1000, embedding_size=200, dimension = 128, wandb_mode='online', PRETRAINED_PATH=None, run_id=None)

# #  60287, 60288, 60289
# main(saving_folder_name=None, additional_name=args.additional_name, criterion='BCELoss', \
#     use_gpu=True, project_name='ICU_lstm_model', experiment='no_pretraining', oversampling=False, \
#             pred_window=2, observing_window=2, BATCH_SIZE=2000, LR=args.lr, min_frequency=5, hidden_size=128,\
#                 drop=0.4, weight_decay=0, num_epochs=1000, embedding_size=150, dimension = 128, wandb_mode='online', PRETRAINED_PATH=None, run_id=None)


# # 60290, 60291, 60292
# main(saving_folder_name=None, additional_name=args.additional_name, criterion='BCELoss', \
#     use_gpu=True, project_name='ICU_lstm_model', experiment='no_pretraining', oversampling=False, \
#     pred_window=2, observing_window=2, BATCH_SIZE=2000, LR=args.lr, min_frequency=5, hidden_size=128,\
#         drop=args.drop, weight_decay=0, num_epochs=1000, embedding_size=args.embedding_size, dimension = 128, \
#             wandb_mode='online', PRETRAINED_PATH=None, run_id=None)


# # 60659, 60661, 60662
# main(saving_folder_name=None, additional_name=args.additional_name, criterion='BCELoss', \
#     use_gpu=True, project_name='ICU_lstm_model', experiment='no_pretraining', oversampling=False, \
#     pred_window=2, observing_window=2, BATCH_SIZE=2000, LR=args.lr, min_frequency=5, hidden_size=128,\
#         drop=args.drop, weight_decay=0, num_epochs=1000, embedding_size=args.embedding_size, dimension = 128, \
#             wandb_mode='online', PRETRAINED_PATH=None, run_id=None)
