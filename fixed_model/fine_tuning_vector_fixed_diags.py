# IMPORTS
from distutils.command.config import config
import pickle5 as pickle
# import pickle

# Libraries
import matplotlib.pyplot as plt
import pandas as pd
import wandb
import os
from os.path import exists

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

# Global variables
global fixed_model_with_diags
global new_fixed_model
global three_stages_model
fixed_model_with_diags = False
new_fixed_model = False
three_stages_model = True

########################################### UTILS ###########################################

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


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):

    if save_path == None:
        return
    
    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_metrics(load_path):

    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

if fixed_model_with_diags:

    def calculate_class_weights(data_loader):
        labels = np.array([])
        for tensor_day, tensor_diags, tensor_labels, idx in data_loader:
            labels = np.concatenate([labels, tensor_labels], axis=0) if labels.size else tensor_labels
        n_pos = np.sum(labels==1, axis=0)
        n_neg = np.sum(labels==0, axis=0)
        pos_weight = np.round(n_neg / n_pos, 2)
        
        return pos_weight
    ########################################### DATASET #################################################
    class MyDataset(Dataset):

        def __init__(self, df, tokenizer, max_length_day=400, pred_window=2, observing_window=3):
            self.df = df
            self.tokenizer = tokenizer
            self.observing_window = observing_window
            self.pred_window = pred_window
            self.max_length_day = max_length_day
            self.max_length_diags = 30

            
        def __len__(self):
            return self.df.shape[0]

        def __getitem__(self, idx):

            return self.make_matrices(idx)
        
        def tokenize(self, text, max_length):
            
            try:
                output = self.tokenizer.encode(text)
            except:
                print(type(text), text, max_length)
                output = self.tokenizer.encode(text)

            # padding and truncation
            if len(output.ids) < max_length:
                len_missing_token = max_length - len(output.ids)
                padding_vec = [self.tokenizer.token_to_id('[PAD]') for _ in range(len_missing_token)]
                token_output = [*output.ids, *padding_vec]
            elif len(output.ids) > max_length:
                token_output = output.ids[:max_length]
            else:
                token_output = output.ids
            
            return token_output

        def make_matrices(self, idx):
            
            day_info = self.df.days_in_visit.values[idx]
            diagnoses_info = self.df.previous_diagnoses.values[idx][0]
            aki_status = self.df.aki_status_in_visit.values[idx]
            days = self.df.days.values[idx]
            # print(idx)

            labels = []
            day_info_list = []
            label = None

            for day in range(days[0], days[0] + self.observing_window + self.pred_window):
                # print('day', day)
                if day not in days:
                    labels.append(0)
                    day_info_list.append(self.tokenize('', self.max_length_day))
                else:
                    i = days.index(day)
                    
                    if np.isfinite(aki_status[i]):                    
                        labels.append(aki_status[i])
                    else:
                        labels.append(0)

                    if (str(day_info[i]) == 'nan') or (day_info[i] == np.nan):
                        day_info_list.append(self.tokenize('[PAD]', self.max_length_day))
                    else:
                        day_info_list.append(self.tokenize(day_info[i], self.max_length_day))


            if sum(labels[-self.pred_window:]) > 0:
                label = 1
            else:
                label = 0

            if (str(diagnoses_info) == 'nan') or (diagnoses_info == np.nan):
                diagnoses_info = self.tokenize('[PAD]', self.max_length_diags)
            else:
                diagnoses_info = self.tokenize(diagnoses_info, self.max_length_diags)

            #make tensors
            tensor_day = torch.tensor(day_info_list[:self.observing_window], dtype=torch.int64)
            tensor_diags = torch.tensor(diagnoses_info, dtype=torch.int64)
            # multilabel
            # tensor_labels = torch.tensor(labels[- self.pred_window:], dtype=torch.float64)
            # one label
            tensor_labels = torch.tensor(label, dtype=torch.float64)
        
            return tensor_day, tensor_diags, tensor_labels, idx

    ########################################### MODEL #################################################

    # Pretraining vector model
    class EHR_PRETRAINING(nn.Module):
        def __init__(self, max_length, vocab_size, device, pred_window=2, observing_window=3,  H=128, embedding_size=200, drop=0.6):
            super(EHR_PRETRAINING, self).__init__()

            self.observing_window = observing_window
            self.pred_window = pred_window
            self.H = H
            self.max_length = max_length
            self.max_length_diags = 30
            self.embedding_size = embedding_size
            self.vocab_size = vocab_size
            self.device = device
            self.drop = drop

            # self.embedding = pretrained_model
            self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)

            self.lstm_day = nn.LSTM(input_size=embedding_size,
                                hidden_size=self.H,
                                num_layers=1,
                                batch_first=True,
                                bidirectional=True)

            self.fc_day = nn.Linear(self.max_length * 2 * self.H, 2048)

            self.fc_adm = nn.Linear(2048*self.observing_window +  self.max_length_diags * 2 * self.H, 2048)

            self.lstm_adm = nn.LSTM(input_size=2048,
                                hidden_size=self.H,
                                num_layers=2,
                                batch_first=True,
                                bidirectional=True)

            self.drop = nn.Dropout(p=drop)
            self.inner_drop = nn.Dropout(p=0.5)

            # self.fc_2 = nn.Linear(self.H*2, 2)
            self.projection = nn.Sequential(
                nn.ReLU(),
                nn.Linear(in_features=self.H*2, out_features=256)
            )

        def forward(self, tensor_day, tensor_diagnoses):

            batch_size = tensor_day.size()[0]

            full_output = torch.tensor([]).to(device=self.device)
            out_emb_diags = self.embedding(tensor_diagnoses.squeeze(1))
            out_lstm_diags, _ = self.lstm_day(out_emb_diags)
            full_output = out_lstm_diags.reshape(batch_size, self.max_length_diags * 2 * self.H)
            

            for d in range(self.observing_window):
                # embedding layer applied to all tensors [16,400,200]
                out_emb = self.embedding(tensor_day[:, d, :].squeeze(1))
                # print('out_emb', out_emb.size())

                # lstm layer applied to embedded tensors
                output_lstm_day= self.inner_drop(self.fc_day(\
                                        self.lstm_day(out_emb)[0]\
                                            .reshape(batch_size, self.max_length * 2 * self.H)))

                # print('output_lstm_day', output_lstm_day.size())                   
                # concatenate for all * days
                full_output = torch.cat([full_output, output_lstm_day], dim=1) # [16, 768]

            # print('full_output size: ', full_output.size(), '\n')
            output = self.fc_adm(full_output)
            # print('output after fc_adm size: ', output.size(), '\n')
            output_vector, _ = self.lstm_adm(output)

            # the fisrt transformation
            output_vector_X = self.drop(output_vector)
            projection_X = self.projection(output_vector_X)
            # the second transformation
            output_vector_Y = self.drop(output_vector)
            projection_Y = self.projection(output_vector_Y)

            return output_vector_X, projection_X, output_vector_Y, projection_Y


    # The fine-tuning model
    class EHR_MODEL(nn.Module):
        def __init__(self, pretrained_model, max_length, vocab_size, device, pred_window=2, observing_window=3,  H=128, embedding_size=200, drop=0.6):
            super(EHR_MODEL, self).__init__()

            self.observing_window = observing_window
            self.pred_window = pred_window
            self.H = H
            self.max_length = max_length
            self.max_length_diags = 30
            self.embedding_size = embedding_size
            self.vocab_size = vocab_size
            self.device = device
            self.drop = drop

            self.embedding = pretrained_model.embedding

            self.lstm_day = pretrained_model.lstm_day

            self.fc_day = pretrained_model.fc_day

            self.fc_adm = pretrained_model.fc_adm

            self.lstm_adm = pretrained_model.lstm_adm

            self.drop = nn.Dropout(p=drop)

            self.fc_2 = nn.Linear(self.H*2, 1)

        def forward(self, tensor_day, tensor_diagnoses):

            batch_size = tensor_day.size()[0]

            # full_output = torch.tensor([]).to(device=self.device)
            out_emb_diags = self.embedding(tensor_diagnoses.squeeze(1))
            out_lstm_diags, _ = self.lstm_day(out_emb_diags)
            full_output = out_lstm_diags.reshape(batch_size, self.max_length_diags * 2 * self.H)
            

            for d in range(self.observing_window):
                # embedding layer applied to all tensors [16,400,200]
                out_emb = self.embedding(tensor_day[:, d, :].squeeze(1))
                # print('out_emb', out_emb.size())

                # lstm layer applied to embedded tensors
                output_lstm_day = self.drop(self.fc_day(\
                                        self.lstm_day(out_emb)[0]\
                                            .reshape(batch_size, self.max_length * 2 * self.H)))

                # print('output_lstm_day', output_lstm_day.size())                   
                # concatenate for all * days
                full_output = torch.cat([full_output, output_lstm_day], dim=1) # [16, 768]

            # print('full_output size: ', full_output.size(), '\n')
            output = self.fc_adm (full_output)
            # print('output after fc_adm size: ', output.size(), '\n')
            output, _ = self.lstm_adm(output)
            # print('output after lstm_adm', output.size())
            output = self.drop(output)
            output = self.fc_2(output)
            # print('output after fc_2', output.size())
            output = torch.squeeze(output, 1)

            # output = nn.Sigmoid()(output)

            return output


    ############################################ FUNCTIONS ###############################################

    def evaluate(model, test_loader, device, threshold=None, log_res=True):
        device = 'cpu'
        model = model.to(device)
        stacked_labels = torch.tensor([]).to(device)
        stacked_probs = torch.tensor([]).to(device)
        
        model.eval()
        step = 1
        with torch.no_grad():
            for tensor_day, tensor_diags, tensor_labels, idx in test_loader:
                # print(f'Step {step}/{len(test_loader)}' )
                labels = tensor_labels.to(device)
                day_info = tensor_day.to(device)
                tensor_diags = tensor_diags.to(device)

                probs = model(day_info, tensor_diags)
                probs = nn.Sigmoid()(probs)
                # output = (probs > threshold).int()

                # stacking labels and predictions
                stacked_labels = torch.cat([stacked_labels, labels], dim=0, )
                # stacked_preds = torch.cat([stacked_preds, output], dim=0, )
                stacked_probs = torch.cat([stacked_probs, probs], dim=0, )
                step += 1
                
        # transfer to device
        stacked_labels = stacked_labels.cpu().detach().numpy()
        stacked_probs = stacked_probs.cpu().detach().numpy()

        if threshold==None:
            if stacked_labels.ndim > 1:
                precision, recall, thresholds = precision_recall_curve(stacked_labels[:].sum(axis=1)>0, np.max(stacked_probs, axis=1))
            else:
                precision, recall, thresholds = precision_recall_curve(stacked_labels, stacked_probs)
                
            # convert to f score
            fscore = np.round((2 * precision * recall) / precision + recall, 2)
            # locate the index of the largest f score
            ix = np.argmax(np.nan_to_num(fscore))
            threshold = np.round(thresholds[ix], 2)
            print('Best Threshold=%.2f, F-Score=%.2f' % (threshold, fscore[ix]))

        stacked_preds = (stacked_probs > threshold).astype(int)
        if stacked_labels.ndim > 1:
            y_true = (stacked_labels[:].sum(axis=1)>0) 
            y_pred = (stacked_preds[:].sum(axis=1)>0)
        else:
            y_true = stacked_labels
            y_pred = stacked_preds

        print(f'The threshold is {threshold}')

        accuracy = np.round(accuracy_score(y_true, y_pred), 2)
        print(f'Accuracy: {accuracy}')

        f1_score_ = np.round(f1_score(y_true, y_pred, pos_label=1, average='binary', zero_division=0), 2)
        print(f'F1: ', f1_score_)

        recall_score_ = np.round(recall_score(y_true, y_pred, pos_label=1, average='binary', zero_division=0), 2)
        print(f'Recall: ', recall_score_)

        precision_score_ = np.round(precision_score(y_true, y_pred, pos_label=1, average='binary', zero_division=0), 2)
        print(f'Precision: ', precision_score_)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity =  np.round(tn / (tn + fp), 2)
        print(f'Specificity: ', specificity)

        print(f'Confusion matrix:\n', confusion_matrix(y_true, y_pred))


        if log_res:
            wandb.log({'test_accuracy':accuracy, 'test_f1_score':f1_score_, 'test_recall_score':recall_score_, 'test_precision_score':precision_score_, 'test_specificity':specificity})

        # get classification metrics for all samples in the test set
        classification_report_res = classification_report(stacked_labels, stacked_preds, zero_division=0, output_dict=True)
        print(classification_report(stacked_labels, stacked_preds, zero_division=0, output_dict=False))

        return {'accuracy':accuracy, 'f1_score':f1_score_, 'recall_score':recall_score_, 'precision_score':precision_score_, 'specificity':specificity}



    def train(model, 
            optimizer,
            train_loader,
            valid_loader,
            file_path,
            device='cpu',
            num_epochs=5,
            criterion = nn.BCELoss(),
            pos_weight = torch.tensor([]),
            best_valid_loss = float("Inf"),
            dimension=128,
            epoch_patience=15,
            threshold=0.5,
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

        sigmoid_fn = nn.Sigmoid()

        if criterion == 'BCEWithLogitsLoss':
            criterion = nn.BCEWithLogitsLoss()
            criterion.pos_weight = pos_weight.to(device)
            use_sigmoid=False
        else:
            criterion = nn.BCELoss()
            use_sigmoid = True

        # training loop
        for epoch in range(num_epochs):  
            stacked_labels = torch.tensor([]).to(device)
            stacked_preds = torch.tensor([]).to(device)

            model.train()
            for tensor_day, tensor_diags, tensor_labels, idx in train_loader:
                # transferring everything to GPU
                labels = tensor_labels.to(device)
                tensor_day = tensor_day.to(device)
                tensor_diags = tensor_diags.to(device)

                output = model(tensor_day, tensor_diags)

                if use_sigmoid:
                    loss = criterion(sigmoid_fn(output), labels.type(torch.float32))
                else:
                    loss = criterion(output, labels.type(torch.float32))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                output_pred = sigmoid_fn(output)
                output_pred = (output_pred > threshold).int()

                stacked_labels = torch.cat([stacked_labels, labels], dim=0)
                stacked_preds = torch.cat([stacked_preds, output_pred], dim=0)
                
                global_step += 1

                wandb.log({'step_train_loss': loss.item(), 'global_step': global_step})
                
            # calculate accuracy
            # epoch_train_accuracy = torch.round(torch.sum(stacked_labels==stacked_preds) / len(stacked_labels), decimals=2)
            if scheduler is not None:
                scheduler.step()
                print(f'Learning rate is {get_lr(optimizer)}')

            model.eval()
            stacked_labels = torch.tensor([]).to(device)
            stacked_preds = torch.tensor([]).to(device)
            with torch.no_grad():
                # validation loop
                for tensor_day, tensor_diags, tensor_labels, idx in valid_loader:
                    labels = tensor_labels.to(device)
                    tensor_day = tensor_day.to(device)
                    tensor_diags = tensor_diags.to(device)
                    
                    output = model(tensor_day, tensor_diags)

                    if use_sigmoid:
                        loss = criterion(sigmoid_fn(output), labels.type(torch.float32))
                    else:
                        loss = criterion(output, labels.type(torch.float32))

                    valid_running_loss += loss.item()

                    output_pred = sigmoid_fn(output)
                    output_pred = (output_pred > threshold).int()

                    # stacking labels and predictions
                    stacked_labels = torch.cat([stacked_labels, labels], dim=0)
                    stacked_preds = torch.cat([stacked_preds, output_pred], dim=0)

            # transfer to device
            stacked_labels = stacked_labels.cpu().detach().numpy()
            stacked_preds = stacked_preds.cpu().detach().numpy()

            # get classification metrics for all samples in the test set
            # classification_report_res = classification_report(stacked_labels, stacked_preds, zero_division=0, output_dict=True)
            # classification_report_res.update({'epoch':epoch+1})

            # log the evaluation metrics 
            # for key, value in classification_report_res.items():
            #     wandb.log({key:value, 'epoch':epoch+1})

            # valid loss
            epoch_average_train_loss = running_loss / len(train_loader)  
            epoch_average_valid_loss = valid_running_loss / len(valid_loader)

            train_loss_list.append(epoch_average_train_loss)
            valid_loss_list.append(epoch_average_valid_loss)
            # train_acc_list.append(epoch_train_accuracy)
            # valid_acc_list.append(epoch_val_accuracy)


            global_steps_list.append(global_step)
            wandb.log({'epoch_average_train_loss': epoch_average_train_loss,
                        'epoch_average_valid_loss': epoch_average_valid_loss,
                        # 'epoch_val_accuracy': epoch_val_accuracy, 
                        # 'epoch_train_accuracy': epoch_train_accuracy,
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

if new_fixed_model:
    def calculate_class_weights(data_loader):
        labels = np.array([])
        for (tensor_demo, tensor_diags, tensor_vitals, tensor_labs, tensor_meds), hadm_id, tensor_labels in data_loader:
            labels = np.concatenate([labels, tensor_labels], axis=0) if labels.size else tensor_labels
        n_pos = np.sum(labels==1, axis=0)
        n_neg = np.sum(labels==0, axis=0)
        pos_weight = np.round(n_neg / n_pos, 2)
        
        return pos_weight
    ########################################### DATASET #################################################
    class MyDataset(Dataset):

        def __init__(self, df, tokenizer, max_length, pred_window=2, observing_window=3):
            self.df = df
            self.tokenizer = tokenizer
            self.observing_window = observing_window
            self.pred_window = pred_window
            self.max_length = max_length
            # self.max_length_diags = 35

            
        def __len__(self):
            return self.df.shape[0]

        def __getitem__(self, idx):

            return self.make_matrices(idx)
        
        def tokenize(self, text, max_length): 
            
            # max_length = max_length + 2
            self.tokenizer.enable_truncation(max_length=max_length)

            output = self.tokenizer.encode(text)

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

        def make_matrices(self, idx):
            
            hadm_id = self.df.hadm_id.values[idx]
            diagnoses_info = self.df.previous_diagnoses.values[idx]
            demo_info = self.df.demographics_in_visit.values[idx][0]
            lab_info = self.df.lab_tests_in_visit.values[idx]
            med_info = self.df.medications_in_visit.values[idx]
            vitals_info = self.df.vitals_in_visit.values[idx]
            
            aki_status = self.df.aki_status_in_visit.values[idx]
            days = self.df.days.values[idx]
            # print(idx)

            lab_info_list = []
            med_info_list = []
            vitals_info_list = []
            labels = []
            label_24 = None
            label_48 = None

            for day in range(days[0], days[0] + self.observing_window + self.pred_window):
                # print('day', day)
                if day not in days:
                    labels.append(0)
                    vitals_info_list.append(self.tokenize('', self.max_length['vitals']))
                    lab_info_list.append(self.tokenize('', self.max_length['lab_tests']))
                    med_info_list.append(self.tokenize('', self.max_length['medications']))

                else:
                    i = days.index(day)

                    if np.isfinite(aki_status[i]):                    
                        labels.append(aki_status[i])
                    else:
                        labels.append(0)

                    # vitals
                    if (str(vitals_info[i]) == 'nan') or (vitals_info[i] == np.nan):
                        vitals_info_list.append(self.tokenize('PAD', self.max_length['vitals']))
                    else:
                        vitals_info_list.append(self.tokenize(vitals_info[i], self.max_length['vitals']))

                    # lab results
                    if (str(lab_info[i]) == 'nan') or (lab_info[i] == np.nan):
                        lab_info_list.append(self.tokenize('PAD', self.max_length['lab_tests']))
                    else:
                        lab_info_list.append(self.tokenize(lab_info[i], self.max_length['lab_tests']))
                    
                    # medications
                    if (str(med_info[i]) == 'nan') or (med_info[i] == np.nan):
                        med_info_list.append(self.tokenize('PAD', self.max_length['medications']))
                    else:
                        med_info_list.append(self.tokenize(med_info[i], self.max_length['medications']))

            # diagnoses
            if (str(diagnoses_info) == 'nan') or (diagnoses_info == np.nan):
                diagnoses_info = self.tokenize('PAD', self.max_length['diagnoses'])
            else:
                diagnoses_info = self.tokenize(diagnoses_info, self.max_length['diagnoses'])

            # demographics
            if (str(demo_info) == 'nan') or (demo_info == np.nan):
                demo_info = self.tokenize('PAD', self.max_length_diags)
            else:
                demo_info = self.tokenize(demo_info, self.max_length['demographics'])

            # get labels for 48h

            # get labels for 24h
            label_24 = labels[-self.pred_window]
            if sum(labels[-self.pred_window:]) > 0:
                label_48 = 1
            else:
                label_48 = 0
            
            #make tensors
            tensor_demo = torch.tensor(demo_info, dtype=torch.int64)
            tensor_diags = torch.tensor(diagnoses_info, dtype=torch.int64)
            tensor_vitals = torch.tensor(vitals_info_list, dtype=torch.int64)
            tensor_labs = torch.tensor(lab_info_list, dtype=torch.int64)
            tensor_meds = torch.tensor(med_info_list, dtype=torch.int64)
            tensor_labels = torch.tensor([label_24, label_48], dtype=torch.float64)
        
            return (tensor_demo, tensor_diags, tensor_vitals, tensor_labs, tensor_meds), hadm_id, tensor_labels


    ########################################### MODEL #################################################

    # Pretraining vector model
    class EHR_PRETRAINING(nn.Module):
        def __init__(self, max_length, vocab_size, device, pred_window=2, observing_window=3,  H=128, embedding_size=200, drop=0.6):
            super(EHR_PRETRAINING, self).__init__()

            self.observing_window = observing_window
            self.pred_window = pred_window
            self.H = H
            self.max_length_demo = max_length['demographics']
            self.max_length_diags = max_length['diagnoses']
            self.max_length_meds = max_length['medications']
            self.max_length_vitals = max_length['vitals']
            self.max_length_lab = max_length['lab_tests']
            self.embedding_size = embedding_size
            self.vocab_size = vocab_size
            self.device = device
            self.drop = drop

            # self.embedding = pretrained_model
            self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)

            self.lstm_day = nn.LSTM(input_size=embedding_size,
                                hidden_size=self.H,
                                num_layers=1,
                                batch_first=True,
                                bidirectional=True)

            self.fc_med = nn.Linear(self.max_length_meds * 2 * self.H, 2048)
            self.fc_vitals = nn.Linear(self.max_length_vitals * 2 * self.H, 2048)
            self.fc_lab = nn.Linear(self.max_length_lab * 2 * self.H, 2048)

            self.fc_adm = nn.Linear(2 * self.H * (self.max_length_diags + self.max_length_demo) \
                                    + self.observing_window * 3 * 2048 , 2048)

            self.lstm_adm = nn.LSTM(input_size=2048,
                                hidden_size=self.H,
                                num_layers=2,
                                batch_first=True,
                                bidirectional=False)

            self.drop = nn.Dropout(p=drop)
            self.inner_drop = nn.Dropout(p=0.5)

            # self.fc_2 = nn.Linear(self.H*2, 2)
            self.projection = nn.Sequential(
                nn.ReLU(),
                nn.Linear(in_features=self.H, out_features=256)
            )

        def forward(self, tensor_demo, tensor_diags, tensor_med, tensor_vitals, tensor_labs):

            batch_size = tensor_demo.size()[0]

            # embeddings
            out_emb_diags = self.embedding(tensor_diags.squeeze(1)) # [16, 37, 200]
            out_emb_demo =  self.embedding(tensor_demo.squeeze(1))  # [16, 7, 200]

            # lstm for demographic and diagnoses
            out_lstm_diags, _ = self.lstm_day(out_emb_diags)    # [16, 37, 256]
            out_lstm_demo, _ = self.lstm_day(out_emb_demo)      # [16, 7, 256]

            # reshape and concat demographics and diags
            out_lstm_diags_reshaped = out_lstm_diags.reshape(batch_size, self.max_length_diags * 2 * self.H)
            out_lstm_demo_reshaped = out_lstm_demo.reshape(batch_size, self.max_length_demo * 2 * self.H)

            full_output = torch.cat([out_lstm_demo_reshaped, out_lstm_diags_reshaped], dim=1)   # [16, 11264]


            for d in range(self.observing_window):

                # embedding layer applied to all tensors 
                out_med_emb = self.embedding(tensor_med[:, d, :].squeeze(1))
                out_vitals_emb = self.embedding(tensor_vitals[:, d, :].squeeze(1))
                out_labs_emb = self.embedding(tensor_labs[:, d, :].squeeze(1))

                # lstm layer applied to embedded tensors
                output_lstm_med = self.inner_drop(self.fc_med(\
                                                    self.lstm_day(out_med_emb)[0]\
                                                        .reshape(batch_size, self.max_length_meds * 2 * self.H)))

                output_lstm_vitals = self.inner_drop(self.fc_vitals(\
                                                    self.lstm_day(out_vitals_emb)[0]\
                                                        .reshape(batch_size, self.max_length_vitals * 2 * self.H)))


                output_lstm_labs = self.inner_drop(self.fc_lab(\
                                                    self.lstm_day(out_labs_emb)[0]\
                                                        .reshape(batch_size, self.max_length_lab * 2 * self.H)))

                            
                # concatenate for all * days
                full_output = torch.cat((full_output, \
                                            output_lstm_med,\
                                                output_lstm_vitals,\
                                                    output_lstm_labs), dim=1) 


            output = self.fc_adm(full_output)

            output_vector, _ = self.lstm_adm(output)

            # the fisrt transformation
            output_vector_X = self.drop(output_vector)
            projection_X = self.projection(output_vector_X)
            # the second transformation
            output_vector_Y = self.drop(output_vector)
            projection_Y = self.projection(output_vector_Y)

            return output_vector_X, projection_X, output_vector_Y, projection_Y

    # The fine-tuning model
    class EHR_FINETUNING(nn.Module):
        def __init__(self, pretrained_model, max_length, vocab_size, device, pred_window=2, observing_window=3,  H=128, embedding_size=200, drop=0.5):
            super(EHR_FINETUNING, self).__init__()

            self.observing_window = observing_window
            self.pred_window = pred_window
            self.H = H
            self.max_length_demo = max_length['demographics']
            self.max_length_diags = max_length['diagnoses']
            self.max_length_meds = max_length['medications']
            self.max_length_vitals = max_length['vitals']
            self.max_length_lab = max_length['lab_tests']
            self.embedding_size = embedding_size
            self.vocab_size = vocab_size
            self.device = device
            self.drop = drop

            # self.embedding = pretrained_model
            self.embedding = pretrained_model.embedding

            self.lstm_day = pretrained_model.lstm_day

            self.fc_med = pretrained_model.fc_med
            self.fc_vitals = pretrained_model.fc_vitals
            self.fc_lab = pretrained_model.fc_lab

            self.fc_adm = pretrained_model.fc_adm

            self.lstm_adm = pretrained_model.lstm_adm

            self.drop = nn.Dropout(p=drop)

            self.fc_2 = nn.Linear(pretrained_model.lstm_adm.hidden_size, 2)

        def forward(self, tensor_demo, tensor_diags, tensor_med, tensor_vitals, tensor_labs):

            batch_size = tensor_demo.size()[0]

            # embeddings
            out_emb_diags = self.embedding(tensor_diags.squeeze(1)) # [16, 37, 200]
            out_emb_demo =  self.embedding(tensor_demo.squeeze(1))  # [16, 7, 200]

            # lstm for demographic and diagnoses
            out_lstm_diags, _ = self.lstm_day(out_emb_diags)    # [16, 37, 256]
            out_lstm_demo, _ = self.lstm_day(out_emb_demo)      # [16, 7, 256]

            # reshape and concat demographics and diags
            out_lstm_diags_reshaped = out_lstm_diags.reshape(batch_size, self.max_length_diags * 2 * self.H)
            out_lstm_demo_reshaped = out_lstm_demo.reshape(batch_size, self.max_length_demo * 2 * self.H)

            full_output = torch.cat([out_lstm_demo_reshaped, out_lstm_diags_reshaped], dim=1)   # [16, 11264]

            for d in range(self.observing_window):
                # embedding layer applied to all tensors 
                out_med_emb = self.embedding(tensor_med[:, d, :].squeeze(1))
                out_vitals_emb = self.embedding(tensor_vitals[:, d, :].squeeze(1))
                out_labs_emb = self.embedding(tensor_labs[:, d, :].squeeze(1))

                # lstm layer applied to embedded tensors
                output_lstm_med = self.drop(self.fc_med(\
                                                    self.lstm_day(out_med_emb)[0]\
                                                        .reshape(batch_size, self.max_length_meds * 2 * self.H)))

                output_lstm_vitals = self.drop(self.fc_vitals(\
                                                    self.lstm_day(out_vitals_emb)[0]\
                                                        .reshape(batch_size, self.max_length_vitals * 2 * self.H)))

                output_lstm_labs = self.drop(self.fc_lab(\
                                                    self.lstm_day(out_labs_emb)[0]\
                                                        .reshape(batch_size, self.max_length_lab * 2 * self.H)))                         
                # concatenate for all * days
                full_output = torch.cat((full_output, \
                                            output_lstm_med,\
                                                output_lstm_vitals,\
                                                    output_lstm_labs), dim=1) # 

            output = self.fc_adm(full_output)
            output, _ = self.lstm_adm(output)
            output = self.drop(output)
            output = self.fc_2(output)
            output = torch.squeeze(output, 1)

            return output


############################################ FUNCTIONS ###############################################

    def evaluate(model, test_loader, device, calculate_threshold=True, threshold=None, log_res=True):
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
            for (tensor_demo, tensor_diags, tensor_vitals, tensor_labs, tensor_meds), hadm_id, tensor_labels in test_loader:
                print(f'Step {step}/{len(test_loader)}' )
                labels = tensor_labels.to(device)
                tensor_demo = tensor_demo.to(device)
                tensor_diags = tensor_diags.to(device)
                tensor_vitals = tensor_vitals.to(device)
                tensor_labs = tensor_labs.to(device)
                tensor_meds = tensor_meds.to(device)

                probs = model(tensor_demo.to(device), tensor_diags.to(device), tensor_meds.to(device), \
                                    tensor_vitals.to(device), tensor_labs.to(device))
                probs = nn.Sigmoid()(probs)
                # output = (probs > threshold).int()

                # stacking labels and predictions
                stacked_labels = torch.cat([stacked_labels, labels], dim=0, )
                # stacked_preds = torch.cat([stacked_preds, output], dim=0, )
                stacked_probs = torch.cat([stacked_probs, probs], dim=0, )
                step += 1
                
        # transfer to device
        stacked_labels = stacked_labels.cpu().detach().numpy()
        stacked_probs = stacked_probs.cpu().detach().numpy()

        for w in range(stacked_labels.ndim):
            pred_window = (w+1)*24
            print('--------------- ', str(pred_window)+'h', '--------------- ')

            precision, recall, thresholds = precision_recall_curve(stacked_labels.T[w], stacked_probs.T[w])
            precision, recall, thresholds = np.round(precision, 2), np.round(recall,2), np.round(thresholds,2)
            
            # convert to f score
            fscore = np.round((2 * precision * recall) / (precision + recall), 2)
            
            if calculate_threshold:
                # locate the index of the largest f score
                ix = np.argmax(np.nan_to_num(fscore))
                threshold = np.round(thresholds[ix], 2)
                print('Best Threshold=%.2f, F-Score=%.2f' % (threshold, fscore[ix]))
            else:
                if threshold==None:
                    print(f'No threshold')

            stacked_preds = (stacked_probs.T[w] > threshold).astype(int)
            y_true = stacked_labels.T[w]
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
            print(f'ROC AUC: ', roc_auc, '\n')
            
            precision_list = [0.2, 0.25, 0.33, 0.4, 0.5, 0.6, 0.75]
            for p in precision_list:
                idx = find_nearest(precision, p)
                sensitivity = recall[idx]
                if idx != len(thresholds):
                    print(f'Precision {np.round(precision[idx]*100, 1)}%, Sensitivity {sensitivity}, Threshold {thresholds[idx]}')
                else:
                    print(f'Precision {np.round(precision[idx]*100, 1)}%, Sensitivity {sensitivity}, Threshold ')

            print(f'Confusion matrix:\n', confusion_matrix(y_true, y_pred))


            if log_res:
                wandb.log({'test_accuracy'+str(pred_window) :accuracy, 'test_f1_score'+str(pred_window):f1_score_, \
                            'test_recall_score'+str(pred_window):recall_score_, 'test_precision_score'+str(pred_window):precision_score_, \
                                'test_specificity'+str(pred_window):specificity})

            # get classification metrics for all samples in the test set
            classification_report_res = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
            print(classification_report(y_true, y_pred, zero_division=0, output_dict=False))

        return stacked_labels, stacked_probs


############################################### TRAIN #################################################
    def train(model, 
            optimizer,
            train_loader,
            valid_loader,
            file_path,
            device='cpu',
            num_epochs=5,
            criterion = nn.BCELoss(),
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

        sigmoid_fn = nn.Sigmoid()

        if criterion == 'BCEWithLogitsLoss':
            criterion = nn.BCEWithLogitsLoss()
            criterion.pos_weight = pos_weight.to(device)
            use_sigmoid=False
        else:
            criterion = nn.BCELoss()
            use_sigmoid = True

        # training loop
        for epoch in range(num_epochs):  
            stacked_labels = torch.tensor([]).to(device)
            stacked_preds = torch.tensor([]).to(device)
            stacked_probs = torch.tensor([]).to(device)

            model.train()
            for (tensor_demo, tensor_diags, tensor_vitals, tensor_labs, tensor_meds), hadm_id, tensor_labels in train_loader:
                # transferring everything to GPU
                labels = tensor_labels.to(device)
                tensor_demo = tensor_demo.to(device)
                tensor_diags = tensor_diags.to(device)
                tensor_vitals = tensor_vitals.to(device)
                tensor_labs = tensor_labs.to(device)
                tensor_meds = tensor_meds.to(device)

                probs = model(tensor_demo.to(device), tensor_diags.to(device), tensor_meds.to(device), \
                                tensor_vitals.to(device), tensor_labs.to(device))

                if use_sigmoid:
                    loss = criterion(sigmoid_fn(probs), labels.type(torch.float32))
                else:
                    loss = criterion(probs, labels.type(torch.float32))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                global_step += 1
                wandb.log({'step_train_loss': loss.item(), 'global_step': global_step})
                
            # calculate accuracy
            # epoch_train_accuracy = torch.round(torch.sum(stacked_labels==stacked_preds) / len(stacked_labels), decimals=2)
            if scheduler is not None:
                scheduler.step()
                print(f'Learning rate is {get_lr(optimizer)}')

            model.eval()
            stacked_labels = torch.tensor([]).to(device)
            stacked_preds = torch.tensor([]).to(device)
            stacked_probs = torch.tensor([]).to(device)

            with torch.no_grad():
                # validation loop
                for (tensor_demo, tensor_diags, tensor_vitals, tensor_labs, tensor_meds), hadm_id, tensor_labels in valid_loader:
                    labels = tensor_labels.to(device)
                    tensor_demo = tensor_demo.to(device)
                    tensor_diags = tensor_diags.to(device)
                    tensor_vitals = tensor_vitals.to(device)
                    tensor_labs = tensor_labs.to(device)
                    tensor_meds = tensor_meds.to(device)
                    
                    probs = model(tensor_demo.to(device), tensor_diags.to(device), tensor_meds.to(device), \
                                tensor_vitals.to(device), tensor_labs.to(device))

                    if use_sigmoid:
                        loss = criterion(sigmoid_fn(probs), labels.type(torch.float32))
                    else:
                        loss = criterion(probs, labels.type(torch.float32))

                    valid_running_loss += loss.item()

                    probs = sigmoid_fn(probs)

                    # stacking labels and predictions
                    stacked_labels = torch.cat([stacked_labels, labels], dim=0)
                    stacked_probs = torch.cat([stacked_probs, probs], dim=0, )

            # transfer to device
            stacked_labels = stacked_labels.cpu().detach().numpy()
            stacked_probs = stacked_probs.cpu().detach().numpy()

            # valid loss
            epoch_average_train_loss = running_loss / len(train_loader)  
            epoch_average_valid_loss = valid_running_loss / len(valid_loader)

            train_loss_list.append(epoch_average_train_loss)
            valid_loss_list.append(epoch_average_valid_loss)
            # train_acc_list.append(epoch_train_accuracy)
            # valid_acc_list.append(epoch_val_accuracy)


            for w in range(stacked_labels.ndim):
                pred_window = (w+1)*24
                precision, recall, thresholds = precision_recall_curve(stacked_labels.T[w], stacked_probs.T[w])
                precision, recall, thresholds = np.round(precision, 2), np.round(recall,2), np.round(thresholds,2)
                
                # convert to f score
                fscore = np.round((2 * precision * recall) / (precision + recall), 2)
                # locate the index of the largest f score
                ix = np.argmax(np.nan_to_num(fscore))
                threshold = np.round(thresholds[ix], 2)
                stacked_preds = (stacked_probs.T[w] > threshold).astype(int)
                y_true = stacked_labels.T[0]
                y_pred = stacked_preds
                f1_score_ = np.round(f1_score(y_true, y_pred, pos_label=1, average='binary', zero_division=0), 2)
                recall_score_ = np.round(recall_score(y_true, y_pred, pos_label=1, average='binary', zero_division=0), 2)
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                specificity =  np.round(tn / (tn + fp), 2)
                pr_auc = np.round(auc(recall, precision), 2)
                wandb.log({'val_f1_score_'+str(pred_window): f1_score_, 'val_recall_score_'+str(pred_window):recall_score_, \
                            'val_specificity'+str(pred_window):specificity, 'val_pr_auc'+str(pred_window):pr_auc,\
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

if three_stages_model:
    class MyDataset(Dataset):

        def __init__(self, df, tokenizer, max_length_day=400, diags='titles', pred_window=2, observing_window=2):
            self.df = df
            self.tokenizer = tokenizer
            self.observing_window = observing_window
            self.pred_window = pred_window
            self.max_length_day = max_length_day
            self.diags = diags

            if self.diags == 'titles':
                self.max_length_diags = 400
            else:
                self.max_length_diags = 40

            
        def __len__(self):
            return self.df.shape[0]

        def __getitem__(self, idx):

            return self.make_matrices(idx)
        
        def tokenize(self, text, max_length):
            
            try:
                output = self.tokenizer.encode(text)
            except:
                # print(idx, type(text), text, max_length)
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

        def make_matrices(self, idx):
            
            hadm_id = self.df.hadm_id.values[idx]
            
            if self.diags == 'titles':
                diagnoses_info = self.df.previous_diags_titles.values[idx][0]
            else:
                diagnoses_info = self.df.previous_diags_icd.values[idx][0]
            
            day_info = self.df.days_in_visit.values[idx]
            days = self.df.days.values[idx]
            AKI_2_status = self.df.AKI_2_in_visit.values[idx]
            AKI_3_status = self.df.AKI_3_in_visit.values[idx]
            # print(hadm_id)
            # print(days)

            AKI_2_labels = []
            AKI_3_labels = []
            day_info_list = []

            AKI_2 = None
            AKI_3 = None

            for day in range(0, self.observing_window + self.pred_window):
                
                if day not in days:
                    # print('not in days')
                    AKI_2_labels.append(0)
                    AKI_3_labels.append(0)
                    day_info_list.append(self.tokenize('', self.max_length_day))
                else:
                    # print('day', day)
                    i = days.index(day)
                    
                    if np.isfinite(AKI_2_status[i]):                    
                        AKI_2_labels.append(AKI_2_status[i])
                    else:
                        AKI_2_labels.append(0)

                    if np.isfinite(AKI_3_status[i]):                    
                        AKI_3_labels.append(AKI_3_status[i])
                    else:
                        AKI_3_labels.append(0)

                    # vitals
                    if (str(day_info[i]) == 'nan') or (day_info[i] == np.nan):
                        day_info_list.append(self.tokenize('PAD', self.max_length_day))
                    else:
                        day_info_list.append(self.tokenize(day_info[i], self.max_length_day))

            # diagnoses
            if (str(diagnoses_info) == 'nan') or (diagnoses_info == np.nan):
                diagnoses_info = self.tokenize('PAD', self.max_length_diags)
            else:
                diagnoses_info = self.tokenize(diagnoses_info, self.max_length_diags)


            if sum(AKI_3_labels[-self.pred_window:]) > 0:
                AKI_2 = 1
                AKI_3 = 1
            elif sum(AKI_2_labels[-self.pred_window:]) > 0:
                AKI_2 = 1
                AKI_3 = 0
            else:
                AKI_2 = 0
                AKI_3 = 0

            #make tensors
            tensor_day = torch.tensor(day_info_list[:self.observing_window], dtype=torch.int64)
            tensor_diags = torch.tensor(diagnoses_info, dtype=torch.int64)
            tensor_labels = torch.tensor([AKI_2, AKI_3], dtype=torch.float64)
        

            return tensor_day, tensor_diags, tensor_labels, hadm_id

################################################ MODEL ################################################
    class EHR_MODEL(nn.Module):
        def __init__(self, pretrained_model, max_length, vocab_size, device, diags='titles', pred_window=2, observing_window=2,  H=128, embedding_size=200, drop=0.6):
            super(EHR_MODEL, self).__init__()

            self.observing_window = observing_window
            self.pred_window = pred_window
            self.H = H
            self.max_length = max_length
            if diags == 'titles':
                self.max_length_diags = 400
            else:
                self.max_length_diags = 4
            self.embedding_size = embedding_size
            self.vocab_size = vocab_size
            self.device = device
            self.drop = drop

            self.embedding = pretrained_model.embedding

            self.lstm_day = pretrained_model.lstm_day

            self.lstm_diags = pretrained_model.lstm_diags

            self.fc_day = pretrained_model.fc_day

            self.fc_adm = pretrained_model.fc_adm

            self.lstm_adm = pretrained_model.lstm_adm

            self.drop = nn.Dropout(p=0.5)

            self.fc_2 = nn.Linear(self.H*2, 2)

        def forward(self, tensor_day, tensor_diagnoses):

            batch_size = tensor_day.size()[0]

            full_output = torch.tensor([]).to(device=self.device)
            out_emb_diags = self.embedding(tensor_diagnoses.squeeze(1))
            # print('out_emb_diags: ', out_emb_diags.size())
            out_lstm_diags, _ = self.lstm_diags(out_emb_diags)
            # print('out_lstm_diags: ', out_lstm_diags.size())
            full_output = out_lstm_diags.reshape(batch_size, self.max_length_diags * 2 * self.H)
            

            for d in range(self.observing_window):
                # embedding layer applied to all tensors [16,400,200]
                out_emb = self.embedding(tensor_day[:, d, :].squeeze(1))
                # print('out_emb', out_emb.size())

                # lstm layer applied to embedded tensors
                output_lstm_day= self.fc_day(\
                                        self.lstm_day(out_emb)[0]\
                                            .reshape(batch_size, self.max_length * 2 * self.H))

                # print('output_lstm_day', output_lstm_day.size())                   
                # concatenate for all * days
                full_output = torch.cat([full_output, output_lstm_day], dim=1) # [16, 768]

            # print('full_output size: ', full_output.size(), '\n')
            output = self.fc_adm (full_output)
            # print('output after fc_adm size: ', output.size(), '\n')
            output, _ = self.lstm_adm(output)
            # print('output after lstm_adm', output.size())
            output = self.drop(output)
            output = self.fc_2(output)
            # print('output after fc_2', output.size())
            output = torch.squeeze(output, 1)

            # output = nn.Sigmoid()(output)

            return output


    class EHR_PRETRAINING(nn.Module):
        def __init__(self, max_length, vocab_size, device, diags='titles', pred_window=2, observing_window=2,  H=128, embedding_size=200, drop=0.6):
            super(EHR_PRETRAINING, self).__init__()

            self.observing_window = observing_window
            self.pred_window = pred_window
            self.H = H
            self.max_length = max_length
            self.drop = drop
            
            if diags == 'titles':
                self.max_length_diags = 400
            else:
                self.max_length_diags = 40

            self.embedding_size = embedding_size
            self.vocab_size = vocab_size
            self.device = device
            self.drop = drop

            # self.embedding = pretrained_model
            self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)

            self.lstm_day = nn.LSTM(input_size=embedding_size,
                                hidden_size=self.H,
                                num_layers=1,
                                batch_first=True,
                                bidirectional=True)

            self.lstm_diags = nn.LSTM(input_size=embedding_size,
                                hidden_size=self.H,
                                num_layers=1,
                                batch_first=True,
                                bidirectional=True)

            self.fc_day = nn.Linear(self.max_length * 2 * self.H, 2048)

            self.fc_adm = nn.Linear(2048*self.observing_window +  self.max_length_diags * 2 * self.H, 2048)

            self.lstm_adm = nn.LSTM(input_size=2048,
                                hidden_size=self.H,
                                num_layers=2,
                                batch_first=True,
                                bidirectional=True)

            self.drop = nn.Dropout(p=drop)
            self.inner_drop = nn.Dropout(p=0.5)

            self.projection = nn.Sequential(
                nn.ReLU(),
                nn.Linear(in_features=2*self.H, out_features=2*256)
            )

        def forward(self, tensor_day, tensor_diagnoses):

            batch_size = tensor_day.size()[0]

            full_output = torch.tensor([]).to(device=self.device)
            out_emb_diags = self.embedding(tensor_diagnoses.squeeze(1))
            # print('out_emb_diags: ', out_emb_diags.size())
            out_lstm_diags, _ = self.lstm_diags(out_emb_diags)
            # print('out_lstm_diags: ', out_lstm_diags.size())
            full_output = out_lstm_diags.reshape(batch_size, self.max_length_diags * 2 * self.H)
            

            for d in range(self.observing_window):
                # embedding layer applied to all tensors [16,400,200]
                out_emb = self.embedding(tensor_day[:, d, :].squeeze(1))
                # print('out_emb', out_emb.size())

                # lstm layer applied to embedded tensors
                output_lstm_day= self.fc_day(\
                                        self.lstm_day(out_emb)[0]\
                                            .reshape(batch_size, self.max_length * 2 * self.H))

                # print('output_lstm_day', output_lstm_day.size())                   
                # concatenate for all * days
                full_output = torch.cat([full_output, output_lstm_day], dim=1) # [16, 768]

            # print('full_output size: ', full_output.size(), '\n')
            output = self.fc_adm (full_output)
            # print('output after fc_adm size: ', output.size(), '\n')
            output_vector, _ = self.lstm_adm(output)
            # the fisrt transformation
            output_vector_X = self.drop(output_vector)
            projection_X = self.projection(output_vector_X)
            # the second transformation
            output_vector_Y = self.drop(output_vector)
            projection_Y = self.projection(output_vector_Y)

            return output_vector_X, projection_X, output_vector_Y, projection_Y

########################################## TRAIN #############################################
    def train(model, 
          optimizer,
          train_loader,
          valid_loader,
          file_path,
          device='cpu',
          num_epochs=5,
          criterion = nn.BCELoss(),
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

        sigmoid_fn = nn.Sigmoid()

        if criterion == 'BCEWithLogitsLoss':
            criterion = nn.BCEWithLogitsLoss()
            criterion.pos_weight = pos_weight.to(device)
            use_sigmoid=False
        else:
            criterion = nn.BCELoss()
            use_sigmoid = True

        # training loop
        for epoch in range(num_epochs):  
            stacked_labels = torch.tensor([]).to(device)
            stacked_preds = torch.tensor([]).to(device)

            model.train()
            for tensor_day, tensor_diags, tensor_labels, idx in train_loader:
                # transferring everything to GPU
                labels = tensor_labels.to(device)
                tensor_day = tensor_day.to(device)
                tensor_diags = tensor_diags.to(device)

                output = model(tensor_day, tensor_diags)

                if use_sigmoid:
                    loss = criterion(sigmoid_fn(output), labels.type(torch.float32))
                else:
                    loss = criterion(output, labels.type(torch.float32))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()            
                global_step += 1
                wandb.log({'step_train_loss': loss.item(), 'global_step': global_step})
                
            # calculate accuracy
            epoch_train_accuracy = torch.round(torch.sum(stacked_labels==stacked_preds) / len(stacked_labels), decimals=2)
            if scheduler is not None:
                scheduler.step()
                print(f'Learning rate is {get_lr(optimizer)}')

            model.eval()
            stacked_labels = torch.tensor([]).to(device)
            stacked_probs = torch.tensor([]).to(device)
            with torch.no_grad():
                # validation loop
                for tensor_day, tensor_diags, tensor_labels, idx in valid_loader:
                    labels = tensor_labels.to(device)
                    tensor_day = tensor_day.to(device)
                    tensor_diags = tensor_diags.to(device)
                    
                    output = model(tensor_day, tensor_diags)

                    if use_sigmoid:
                        loss = criterion(sigmoid_fn(output), labels.type(torch.float32))
                    else:
                        loss = criterion(output, labels.type(torch.float32))

                    valid_running_loss += loss.item()
                    probs = sigmoid_fn(output)
                    # stacking labels and predictions
                    stacked_labels = torch.cat([stacked_labels, labels], dim=0)
                    stacked_probs = torch.cat([stacked_probs, probs], dim=0, )

            # transfer to device
            stacked_labels = stacked_labels.cpu().detach().numpy()
            stacked_probs = stacked_probs.cpu().detach().numpy()
            # valid loss
            epoch_average_train_loss = running_loss / len(train_loader)  
            epoch_average_valid_loss = valid_running_loss / len(valid_loader)

            train_loss_list.append(epoch_average_train_loss)
            valid_loss_list.append(epoch_average_valid_loss)
            stages = ['AKI 2', 'AKI 2,3']
            for w in range(stacked_labels.ndim):
                stage = stages[w]
                precision, recall, thresholds = precision_recall_curve(stacked_labels.T[w], stacked_probs.T[w])
                precision, recall, thresholds = np.round(precision, 2), np.round(recall,2), np.round(thresholds,2)
                
                # convert to f score
                fscore = np.round((2 * precision * recall) / (precision + recall), 2)
                # locate the index of the largest f score
                ix = np.argmax(np.nan_to_num(fscore))
                threshold = np.round(thresholds[ix], 2)
                stacked_preds = (stacked_probs.T[w] > threshold).astype(int)
                y_true = stacked_labels.T[0]
                y_pred = stacked_preds
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

        print('Finished Training!')

    def evaluate(model, test_loader, device, threshold=None, log_res=True):
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
            for tensor_day, tensor_diags, tensor_labels, idx in test_loader:
                print(f'Step {step}/{len(test_loader)}' )

                labels = tensor_labels.to(device)
                day_info = tensor_day.to(device)
                tensor_diags = tensor_diags.to(device)

                probs = model(day_info, tensor_diags)
                probs = nn.Sigmoid()(probs)
                # output = (probs > threshold).int()

                # stacking labels and predictions
                stacked_labels = torch.cat([stacked_labels, labels], dim=0, )
                # stacked_preds = torch.cat([stacked_preds, output], dim=0, )
                stacked_probs = torch.cat([stacked_probs, probs], dim=0, )
                step += 1
                
        # transfer to device
        stacked_labels = stacked_labels.cpu().detach().numpy()
        stacked_probs = stacked_probs.cpu().detach().numpy()
        # for printing purposes
        stages_names = ['2', '2 and 3']
        if threshold==None:
            for w in range(stacked_labels.ndim):
                print('------------- AKI stage ', stages_names[w], '------------- ')

                precision, recall, thresholds = precision_recall_curve(stacked_labels.T[w], stacked_probs.T[w])
                precision, recall, thresholds = np.round(precision, 2), np.round(recall,2), np.round(thresholds,2)
                
                # convert to f score
                fscore = np.round((2 * precision * recall) / (precision + recall), 2)

                # locate the index of the largest f score
                ix = np.argmax(np.nan_to_num(fscore))
                threshold = np.round(thresholds[ix], 2)
                print('Best Threshold=%.2f, F-Score=%.2f' % (threshold, fscore[ix]))

                stacked_preds = (stacked_probs.T[w] > threshold).astype(int)
                y_true = stacked_labels.T[w]
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

######################## MAIN ###############################
def main(saving_folder_name=None, additional_name='', criterion='BCELoss', pos_weight=None, \
    small_dataset=False, use_gpu=True, project_name='test', experiment='test', oversampling=False, \
        diagnoses='icd', pred_window=2,  observing_window=2, weight_decay=0, BATCH_SIZE=128, \
            LR=0.0001, min_frequency=1, hidden_size=128, drop=0.6, num_epochs=50, wandb_mode='online', \
                PRETRAINED_PATH=None, run_id=None, checkpoint=None):
    # define the device
    if use_gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device='cpu'

    #paths
    CURR_PATH = os.getcwd()
    print(f'Current working directory is {CURR_PATH}')
    PKL_PATH = CURR_PATH+'/pickles/'
    DF_PATH = CURR_PATH +'/dataframes/'

    destination_folder = '/l/users/svetlana.maslenkova/models' + '/three_stages_model/fine_tuning/'
    # destination_folder = '/home/svetlanamaslenkova/Documents/AKI_deep/LSTM/training/'
    
    if diagnoses=='icd':
        TOKENIZER_PATH = CURR_PATH + '/aki_prediction/' + '/tokenizer.json'
        TXT_DIR_TRAIN = CURR_PATH + '/aki_prediction/' + '/txt_files/train'
    elif diagnoses=='titles':
        TOKENIZER_PATH = CURR_PATH + '/aki_prediction/'+ '/tokenizer_titles.json'
        TXT_DIR_TRAIN = CURR_PATH + '/aki_prediction/'+ '/txt_files/titles_diags'

    # Training the tokenizer
    if exists(TOKENIZER_PATH):
        tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
        print(f'Tokenizer is loaded from ==> {TOKENIZER_PATH}/tokenizer.json. Vocab size is {tokenizer.get_vocab_size()}')
    else:
        print('Training tokenizer...')
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        tokenizer = Tokenizer(BPE(unk_token="UNK"))
        tokenizer.normalizer = normalizers.Sequence([Lowercase()])
        # tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Digits(individual_digits=False), Punctuation( behavior = 'removed')])
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Punctuation(behavior = 'isolated')])

        trainer = BpeTrainer(special_tokens=["<s>", "</s>", "PAD", "UNK", "$"], min_frequency=min_frequency)

        files = glob.glob(TXT_DIR_TRAIN+'/*')
        tokenizer.train(files, trainer)
        tokenizer.post_processor = BertProcessing(
                ("</s>", tokenizer.token_to_id("</s>")),
                ("<s>", tokenizer.token_to_id("<s>")), 
                )
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        print(f'Vocab size is {tokenizer.get_vocab_size()}')

    # variables for classes
    if fixed_model_with_diags or three_stages_model:
        max_length = 400
    elif new_fixed_model:
        max_length = {'demographics':5+2, 'diagnoses':35+2, 'lab_tests':300+2, 'vitals':31+2, 'medications':256+2}
    vocab_size = tokenizer.get_vocab_size()
    embedding_size = 200
    dimension = 128
    

    # loading the data
    with open(DF_PATH + 'pid_train_df_finetuning.pkl', 'rb') as f:
        pid_train_df = pickle.load(f)

    with open(DF_PATH + 'pid_val_df_finetuning.pkl', 'rb') as f:
        pid_val_df = pickle.load(f)

    with open(DF_PATH + 'pid_test_df_finetuning.pkl', 'rb') as f:
        pid_test_df = pickle.load(f)


    # pid_train_df = pid_train_df[pid_train_df.hadm_id.isin(train_admissions)]
    # pid_val_df = pid_val_df[pid_val_df.hadm_id.isin(val_admissions)]
    # pid_test_df = pid_test_df[pid_test_df.hadm_id.isin(test_admissions)]

    if small_dataset: frac=0.1
    else: frac=1

    if fixed_model_with_diags:
        train_dataset = MyDataset(pid_train_df.sample(frac=frac), tokenizer=tokenizer, max_length_day=400)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        val_dataset = MyDataset(pid_val_df.sample(frac=frac), tokenizer=tokenizer, max_length_day=400)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

        test_dataset = MyDataset(pid_test_df.sample(frac=frac), tokenizer=tokenizer, max_length_day=400)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    elif new_fixed_model:
        train_dataset = MyDataset(pid_train_df.sample(frac=frac), tokenizer=tokenizer, max_length=max_length)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        test_dataset = MyDataset(pid_test_df.sample(frac=frac), tokenizer=tokenizer, max_length=max_length)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

        val_dataset = MyDataset(pid_val_df.sample(frac=frac), tokenizer=tokenizer, max_length=max_length)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    elif three_stages_model:
        train_dataset = MyDataset(pid_train_df.sample(frac=frac), tokenizer, max_length_day=400, diags='titles', pred_window=2, observing_window=2)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        val_dataset = MyDataset(pid_val_df.sample(frac=frac), tokenizer, max_length_day=400, diags='titles', pred_window=2, observing_window=2)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

        test_dataset = MyDataset(pid_test_df.sample(frac=frac), tokenizer, max_length_day=400, diags='titles', pred_window=2, observing_window=2)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    if oversampling:
        i = 0
        for tensor_day, tensor_diags, tensor_labels, hadm_id in train_loader:
            if i == 0:stacked_labels = tensor_labels
            else:stacked_labels = np.concatenate([stacked_labels, tensor_labels])
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
    #print shapes
    # tensor_day, tensor_labels, = next(iter(train_loader))
    # tensor_day, tensor_diags, tensor_labels, idx = next(iter(train_loader))
    print('\n\n DATA SHAPES: ')
    print('train data shape: ', len(train_dataset))
    print('val data shape: ', len(val_dataset))
    print('test data shape: ', len(val_dataset))


    ## load pretrained model
    pretrained_model =  EHR_PRETRAINING(max_length, vocab_size, device, diags='titles', pred_window=2, observing_window=2,  H=hidden_size, embedding_size=200, drop=drop).to(device)
    if PRETRAINED_PATH is not None:
        pretrained_model.load_state_dict(torch.load(PRETRAINED_PATH, map_location=device)['model_state_dict'])
        print(f"Pretrained model loaded from <=== {PRETRAINED_PATH}")

    if fixed_model_with_diags:
        model = EHR_MODEL(pretrained_model, max_length, vocab_size, device, pred_window=2, observing_window=3,  H=hidden_size, embedding_size=200, drop=drop).to(device)
    elif new_fixed_model:
        model = EHR_FINETUNING(pretrained_model, max_length, vocab_size, device, drop=drop).to(device)
    elif three_stages_model:
        model = EHR_MODEL(pretrained_model, max_length, vocab_size, device, diags='titles', pred_window=2, observing_window=2,  H=hidden_size, embedding_size=200, drop=drop)
    
    for num, (name, param) in enumerate(model.named_parameters()):
        if num < 21:
            if experiment == 'no_pretraining':
                param.requires_grad = True
            else:
                param.requires_grad = False
            print(f'Parameter {num}: {name}.requires_grad == {param.requires_grad}')
        else:
            param.requires_grad = True
            print(f'Parameter {num}: {name}.requires_grad == {param.requires_grad}')

    ## create optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=weight_decay)
    ## Decay LR by a factor of 0.1 every 7 epochs
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[15,30,50], gamma=0.1)
    exp_lr_scheduler = None

    ## efine training parameters
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
    use_sigmoid = True

    if criterion=='BCEWithLogitsLoss':
        #calculate weights
        print('Calculating class weights..')
        if pos_weight==None:
            pos_weight = torch.tensor(calculate_class_weights(train_loader))
        print(f'Calss weights are {pos_weight}')
        pos_weight = torch.tensor(pos_weight, dtype=torch.int64).to(device)
        train_params['pos_weight'] = pos_weight
        weights = 'with_weights'
        use_sigmoid = False

    # path for saving the model
    if saving_folder_name is None:
        saving_folder_name = additional_name + 'FT_' + experiment + str(len(train_dataset) // 1000) + 'k_'  \
            + 'lr' + str(LR) + '_h'+ str(hidden_size) + '_pw' + str(pred_window) + '_ow' + str(observing_window) \
                + '_wd' + str(weight_decay) + '_'+ weights + '_drop' + str(drop)
    
    file_path = destination_folder + saving_folder_name
    train_params['file_path'] = file_path

    print(f'\n\nMODEL PATH: {file_path}')
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

    if checkpoint is not None:
        load_checkpoint(checkpoint, model, optimizer, device=device)

        
    args = {'optimizer':optimizer, 'criterion':'BCELoss', 'LR':LR, 'min_frequency':min_frequency, 'hidden_size':hidden_size, \
            'pred_window':pred_window, 'experiment':experiment, 'weight_decay':weight_decay, 'drop':drop}
    wandb.init(project=project_name, name=run_name, mode=wandb_mode, config=args, id=run_id, resume=resume)
    print('Run id is: ', run_id)

    # training
    print('Training started..')
    train(**train_params)

    # testing
    print('\nTesting the model...')
    load_checkpoint(file_path + '/model.pt', model, optimizer, device=device)
    evaluate(model, test_loader, device, threshold=None, log_res=True)

    wandb.finish()


# #paths
# print('Filtering admissions...')
# CURR_PATH = os.getcwd()
# PKL_PATH = CURR_PATH+'/pickles/'
# DF_PATH = CURR_PATH +'/dataframes/'

# # loading the data
# with open(DF_PATH + 'pid_train_df_finetuning_6days_aki.pkl', 'rb') as f:
#     pid_train_df = pickle.load(f)

# with open(DF_PATH + 'pid_val_df_finetuning_6days_aki.pkl', 'rb') as f:
#     pid_val_df = pickle.load(f)

# with open(DF_PATH + 'pid_test_df_finetuning_6days_aki.pkl', 'rb') as f:
#     pid_test_df = pickle.load(f)

# observing_window = 3 

# train_admissions = []
# for adm in pid_train_df.hadm_id.unique():   
#     if ({1,2,3,4}.issubset(set(pid_train_df[pid_train_df.hadm_id==adm].days.values[0])) or \
#         {-1,0,1,2}.issubset(set(pid_train_df[pid_train_df.hadm_id==adm].days.values[0]))or \
#             {0,1,2,3}.issubset(set(pid_train_df[pid_train_df.hadm_id==adm].days.values[0]))) and \
#         (len(pid_train_df[pid_train_df.hadm_id==adm].days.values[0])>3) and\
#             sum(pid_train_df[pid_train_df.hadm_id==adm].aki_status_in_visit.values[0][:observing_window])==0:
#         train_admissions.append(adm)

# val_admissions = []
# for adm in pid_val_df.hadm_id.unique():   
#     if ({1,2,3,4}.issubset(set(pid_val_df[pid_val_df.hadm_id==adm].days.values[0])) or \
#         {-1,0,1,2}.issubset(set(pid_val_df[pid_val_df.hadm_id==adm].days.values[0]))or \
#             {0,1,2,3}.issubset(set(pid_val_df[pid_val_df.hadm_id==adm].days.values[0]))) and \
#         (len(pid_val_df[pid_val_df.hadm_id==adm].days.values[0])>3) and\
#             sum(pid_val_df[pid_val_df.hadm_id==adm].aki_status_in_visit.values[0][:observing_window])==0:
#         val_admissions.append(adm)

# test_admissions = []
# for adm in pid_test_df.hadm_id.unique():   
#     if ({1,2,3,4}.issubset(set(pid_test_df[pid_test_df.hadm_id==adm].days.values[0])) or \
#         {-1,0,1,2}.issubset(set(pid_test_df[pid_test_df.hadm_id==adm].days.values[0]))or \
#             {0,1,2,3}.issubset(set(pid_test_df[pid_test_df.hadm_id==adm].days.values[0]))) and \
#         (len(pid_test_df[pid_test_df.hadm_id==adm].days.values[0])>3) and\
#             sum(pid_test_df[pid_test_df.hadm_id==adm].aki_status_in_visit.values[0][:observing_window])==0:
#         test_admissions.append(adm)

# print('train_admissions', len(train_admissions))
# print('val_admissions', len(val_admissions))
# print('test_admissions', len(test_admissions))


################################################# RUNs ####################################################

## test run
# PRETRAINED_PATH = '/l/users/svetlana.maslenkova/models/pretraining/fc1_fixed/CL_WHOLE_FX__bs64_376k_lr1e-05_Adam_temp0.05_drop0.2/model.pt'
# main(saving_folder_name='test_model', criterion='BCELoss', small_dataset=True,\
#      use_gpu=False, project_name='test', pred_window=2, weight_decay=0, BATCH_SIZE=128  , LR=1e-05,\
#          min_frequency=5, hidden_size=128, drop=0.4, num_epochs=1, wandb_mode='disabled', PRETRAINED_PATH=PRETRAINED_PATH, run_id=None)

# ## 37166: not filtered dataset
# PRETRAINED_PATH = '/l/users/svetlana.maslenkova/models/pretraining/fc1_fixed/CL_WHOLE_FX__bs64_376k_lr1e-05_Adam_temp0.05_drop0.2/model.pt'
# main(saving_folder_name=None, additional_name='_temp0.05_drop0.2_', criterion='BCELoss', small_dataset=False,\
#      use_gpu=True, project_name='Fixed_obs_window_model', pred_window=2, weight_decay=0, BATCH_SIZE=128  , LR=1e-05,\
#          min_frequency=5, hidden_size=128, drop=0.4, num_epochs=100, wandb_mode='online', PRETRAINED_PATH=PRETRAINED_PATH, run_id=None)

# # 37177
# PRETRAINED_PATH = '/l/users/svetlana.maslenkova/models/pretraining/fc1_fixed/CL_WHOLE_FX__bs64_376k_lr1e-05_Adam_temp0.05_drop0.2/model.pt'
# main(saving_folder_name=None, additional_name='_temp0.05_drop0.2_', criterion='BCELoss', small_dataset=False,\
#      use_gpu=True, project_name='Fixed_obs_window_model', pred_window=2, weight_decay=0, BATCH_SIZE=128  , LR=1e-05,\
#          min_frequency=5, hidden_size=128, drop=0.4, num_epochs=100, wandb_mode='online', PRETRAINED_PATH=PRETRAINED_PATH, run_id=None)

# # 37182, continue 37204
# PRETRAINED_PATH = '/l/users/svetlana.maslenkova/models/pretraining/fc1_fixed/CL_WHOLE_FX__bs64_376k_lr1e-05_Adam_temp0.1_drop0.2/model.pt'
# checkpoint = '/l/users/svetlana.maslenkova/models/finetuning/embeddings/FT_PRE_WHOLE_D__temp0.1_drop0.2_10k_lr1e-05_h128_pw2_ow3_wd0__drop0.4/model.pt'
# main(saving_folder_name=None, additional_name='_temp0.1_drop0.2_', criterion='BCELoss', small_dataset=False,\
#      use_gpu=True, project_name='Fixed_obs_window_model', pred_window=2, weight_decay=0, BATCH_SIZE=128  , LR=1e-05,\
#          min_frequency=5, hidden_size=128, drop=0.4, num_epochs=500, wandb_mode='online', PRETRAINED_PATH=PRETRAINED_PATH, run_id=None, checkpoint=checkpoint)

# # # 37183, continue training 37203, 37205
# PRETRAINED_PATH = '/l/users/svetlana.maslenkova/models/pretraining/fc1_fixed/CL_WHOLE_FX__bs64_376k_lr1e-05_Adam_temp0.05_drop0.1/model.pt'
# checkpoint = '/l/users/svetlana.maslenkova/models/finetuning/embeddings/FT_PRE_WHOLE_D__temp0.05_drop0.1_10k_lr1e-05_h128_pw2_ow3_wd0__drop0.4/model.pt'
# main(saving_folder_name=None, additional_name='_temp0.05_drop0.1_', criterion='BCELoss', small_dataset=False,\
#      use_gpu=True, project_name='Fixed_obs_window_model', pred_window=2, weight_decay=0, BATCH_SIZE=128  , LR=1e-05,\
#          min_frequency=5, hidden_size=128, drop=0.4, num_epochs=500, wandb_mode='online', PRETRAINED_PATH=PRETRAINED_PATH, run_id=None, checkpoint=checkpoint)

# # # # 37184, continue training 37201, 37206
# PRETRAINED_PATH = '/l/users/svetlana.maslenkova/models/pretraining/fc1_fixed/CL_WHOLE_FX__bs64_376k_lr1e-05_Adam_temp0.1_drop0.1/model.pt'
# checkpoint = '/l/users/svetlana.maslenkova/models/finetuning/embeddings/FT_PRE_WHOLE_D__temp0.1_drop0.1_10k_lr1e-05_h128_pw2_ow3_wd0__drop0.4/model.pt'
# main(saving_folder_name=None, additional_name='_temp0.1_drop0.1_', criterion='BCELoss', small_dataset=False,\
#      use_gpu=True, project_name='Fixed_obs_window_model', pred_window=2, weight_decay=0, BATCH_SIZE=128  , LR=1e-05,\
#          min_frequency=5, hidden_size=128, drop=0.4, num_epochs=500, wandb_mode='online', PRETRAINED_PATH=PRETRAINED_PATH, run_id=None, checkpoint=checkpoint)

############################################## NEW MODEL ###########################################################
# ## test run
# PRETRAINED_PATH = '/l/users/svetlana.maslenkova/models/pretraining/fc1_fixed/CL_WHOLE_FX_ND__bs800_376k_lr1e-05_Adam_temp0.1_drop0.1/model.pt'
# main(saving_folder_name='test_model', criterion='BCELoss', small_dataset=True,\
#      use_gpu=False, project_name='test', pred_window=2, weight_decay=0, BATCH_SIZE=128  , LR=1e-05,\
#          min_frequency=5, hidden_size=128, drop=0.4, num_epochs=1, wandb_mode='disabled', PRETRAINED_PATH=PRETRAINED_PATH, run_id=None)

# ## 39213
# PRETRAINED_PATH = None
# main(saving_folder_name=None, criterion='BCELoss', small_dataset=False,\
#      use_gpu=True, project_name='Fixed_obs_window_model', experiment='no_pretraining', pred_window=2, weight_decay=0, BATCH_SIZE=1024  , LR=1e-05,\
#          min_frequency=10, hidden_size=128, drop=0.5, num_epochs=100, wandb_mode='online', PRETRAINED_PATH=PRETRAINED_PATH, run_id=None)

# ## 39216
# PRETRAINED_PATH = '/l/users/svetlana.maslenkova/models/pretraining/fc1_fixed/CL_WHOLE_FX_ND__bs800_376k_lr1e-05_Adam_temp0.1_drop0.1/model.pt'
# main(saving_folder_name=None, criterion='BCELoss', small_dataset=False,\
#      use_gpu=True, project_name='Fixed_obs_window_model', experiment='pretrained', pred_window=2, weight_decay=0, BATCH_SIZE=1024  , LR=1e-05,\
#          min_frequency=10, hidden_size=128, drop=0.5, num_epochs=100, wandb_mode='online', PRETRAINED_PATH=PRETRAINED_PATH, run_id=None)

# # # 39236(wrond labels), 39456, 39585
# PRETRAINED_PATH = None
# main(saving_folder_name=None, experiment='no_pretraining', criterion='BCELoss', small_dataset=False,\
#      use_gpu=True, project_name='Fixed_obs_window_model', pred_window=2, weight_decay=0, BATCH_SIZE=2048  , LR=1e-05,\
#          min_frequency=10, hidden_size=128, drop=0.5, num_epochs=150, wandb_mode='online', PRETRAINED_PATH=PRETRAINED_PATH, run_id=None)

# # ### 39237(wrond labels), 39458, 39583
# PRETRAINED_PATH = '/l/users/svetlana.maslenkova/models/pretraining/fc1_fixed/CL_WHOLE_FX_ND__bs800_376k_lr1e-05_Adam_temp0.1_drop0.1/model.pt'
# main(saving_folder_name=None, experiment='PRE', criterion='BCELoss', small_dataset=False,\
#      use_gpu=True, project_name='Fixed_obs_window_model',  pred_window=2, weight_decay=0, BATCH_SIZE=2048  , LR=1e-05,\
#          min_frequency=10, hidden_size=128, drop=0.5, num_epochs=150, wandb_mode='online', PRETRAINED_PATH=PRETRAINED_PATH, run_id=None)


# # # ### with weights 39832, 40587
# PRETRAINED_PATH = '/l/users/svetlana.maslenkova/models/pretraining/fc1_fixed/CL_WHOLE_FX_ND__bs800_376k_lr1e-05_Adam_temp0.1_drop0.1/model.pt'
# main(saving_folder_name=None, experiment='PRE', criterion='BCEWithLogitsLoss', pos_weight=[4.7056928 , 2.28509586], small_dataset=False,\
#      use_gpu=True, project_name='Fixed_obs_window_model',  pred_window=2, weight_decay=0, BATCH_SIZE=1024  , LR=1e-05,\
#          min_frequency=10, hidden_size=128, drop=0.5, num_epochs=150, wandb_mode='online', PRETRAINED_PATH=PRETRAINED_PATH, run_id=None)

# # # with weights 39833(forgot to infreeze layers), 40554, 40586
# PRETRAINED_PATH = None
# main(saving_folder_name=None, experiment='no_pretraining', criterion='BCEWithLogitsLoss',  pos_weight=[4.7056928 , 2.28509586], small_dataset=False,\
#      use_gpu=True, project_name='Fixed_obs_window_model', pred_window=2, weight_decay=0, BATCH_SIZE=1024  , LR=1e-05,\
#          min_frequency=10, hidden_size=128, drop=0.5, num_epochs=150, wandb_mode='online', PRETRAINED_PATH=PRETRAINED_PATH, run_id=None)

# # # # ### with weights 39834
# PRETRAINED_PATH = '/l/users/svetlana.maslenkova/models/pretraining/fc1_fixed/CL_WHOLE_FX_ND__bs800_376k_lr1e-05_Adam_temp0.1_drop0.1/model.pt'
# main(saving_folder_name=None, experiment='PRE', criterion='BCEWithLogitsLoss', pos_weight=[4.7056928 , 2.28509586], small_dataset=False,\
#      use_gpu=True, project_name='Fixed_obs_window_model',  pred_window=2, weight_decay=0, BATCH_SIZE=1024  , LR=1e-04,\
#          min_frequency=10, hidden_size=128, drop=0.5, num_epochs=150, wandb_mode='online', PRETRAINED_PATH=PRETRAINED_PATH, run_id=None)

# # # # # with weights 39835(forgot to infreeze layers), 40546, 
# PRETRAINED_PATH = None
# main(saving_folder_name=None, experiment='no_pretraining', criterion='BCEWithLogitsLoss',  pos_weight=[4.7056928 , 2.28509586], small_dataset=False,\
#      use_gpu=True, project_name='Fixed_obs_window_model', pred_window=2, weight_decay=0, BATCH_SIZE=1024  , LR=1e-04,\
#          min_frequency=10, hidden_size=128, drop=0.5, num_epochs=150, wandb_mode='online', PRETRAINED_PATH=PRETRAINED_PATH, run_id=None)



##############################################################################################################
                            ################## THREE STAGES MODEL ##################
##############################################################################################################

# # test run
PRETRAINED_PATH = '/l/users/svetlana.maslenkova/models/pretraining/three_stages/STG_bs300_390k_lr0.0001_Adam_temp0.5_drop0.1/model.pt'
main(saving_folder_name='test_model', additional_name='', criterion='BCELoss', pos_weight=None, \
    small_dataset=True, use_gpu=False, project_name='test', experiment='test', oversampling=False, \
        diagnoses='titles', pred_window=2,  observing_window=2, weight_decay=0, BATCH_SIZE=128, \
            LR=0.00001, min_frequency=10, hidden_size=128, drop=0.6, num_epochs=1, wandb_mode='disabled', \
                PRETRAINED_PATH=PRETRAINED_PATH, run_id=None, checkpoint=None)

