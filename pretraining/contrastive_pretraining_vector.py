# IMPORTS
import torch
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tokenizers import  Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Punctuation, Whitespace
from tokenizers.normalizers import Lowercase
from tokenizers import pre_tokenizers, normalizers
from tokenizers.processors import BertProcessing

import glob
from os.path import exists
import os

import pickle5 as pickle
import wandb

import pandas as pd
import numpy as np

# Global variables
global fixed_model_with_diags
global cont_model
global new_fixed_model
global model_three_stages
fixed_model_with_diags = False
cont_model = False
new_fixed_model = False
model_three_stages = True


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.device = device
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())
            
    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)      
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask.to(self.device) * torch.exp(similarity_matrix / self.temperature)
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss


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



if fixed_model_with_diags:

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
            # aki_status = self.df.aki_status_in_visit.values[idx]
            days = self.df.days.values[idx]
            # print(idx)

            day_info_list = []
            label = None

            for day in range(days[0], days[0] + self.observing_window + self.pred_window):
                # print('day', day)
                if day not in days:
                    
                    day_info_list.append(self.tokenize('', self.max_length_day))
                else:
                    i = days.index(day)
                    

                    if (str(day_info[i]) == 'nan') or (day_info[i] == np.nan):
                        day_info_list.append(self.tokenize('[PAD]', self.max_length_day))
                    else:
                        day_info_list.append(self.tokenize(day_info[i], self.max_length_day))


            if (str(diagnoses_info) == 'nan') or (diagnoses_info == np.nan):
                diagnoses_info = self.tokenize('[PAD]', self.max_length_diags)
            else:
                diagnoses_info = self.tokenize(diagnoses_info, self.max_length_diags)

            #make tensors
            tensor_day = torch.tensor(day_info_list[:self.observing_window], dtype=torch.int64)
            tensor_diags = torch.tensor(diagnoses_info, dtype=torch.int64)
            # tensor_labels = torch.tensor(label, dtype=torch.float64)
        
            return tensor_day, tensor_diags, idx


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
                                bidirectional=False)

            self.drop = nn.Dropout(p=drop)
            self.inner_drop = nn.Dropout(p=0.5)

            # self.fc_2 = nn.Linear(self.H*2, 2)
            self.projection = nn.Sequential(
                nn.ReLU(),
                nn.Linear(in_features=self.H, out_features=256)
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

    def train(model,
            optimizer,
            train_loader,
            valid_loader,
            batch_size,
            file_path,
            embedding_size,
            device,
            num_epochs=2,
            epoch_patience=10,
            best_valid_loss = float("Inf"),
            dimension=128,
            save_model=True,
            temperature=0.1):

        
        train_running_loss = 0.0
        valid_running_loss = 0.0
        train_loss_list = []
        valid_loss_list = []
        total_train_steps = len(train_loader)
        total_val_steps = len(valid_loader)
        stop_training = 0


        for epoch in range(num_epochs):
                model.train()
                train_step = 1
                print(f'Epoch {epoch+1}/{num_epochs} training..')
                criterion = ContrastiveLoss(batch_size=batch_size, device=device, temperature=temperature)

                for  tensor_day, tensor_diags, idx in train_loader:
                    tensor_day = tensor_day.to(device)
                    tensor_diags = tensor_diags.to(device)

                    if train_step % 100==0:
                        print(f'Step {train_step}/{total_train_steps}')

                    d = 0                
                    vector_X, projectionX, vector_Y, projectionY = model(tensor_day, tensor_diags)
                                            
                    if train_step >= total_train_steps:
                            new_batch_size = projectionX.size()[0]
                            criterion = ContrastiveLoss(batch_size=new_batch_size, device=device, temperature=temperature)
                    
                    loss = criterion(projectionX.type(torch.float32), projectionY.type(torch.float32))

                    #   print(loss.item())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_running_loss += loss.item()
                    train_step += 1

                    wandb.log({'step_train_loss': loss.item(), 'global_step': train_step})

                epoch_average_train_loss = train_running_loss / len(train_loader)  

                model.eval()
                val_step = 1
                print(f"Validation started..")
                criterion = ContrastiveLoss(batch_size=batch_size, device=device, temperature=temperature)
                with torch.no_grad():
                    for  tensor_day, tensor_diags, idx in valid_loader:
                            tensor_day = tensor_day.to(device)
                            tensor_diags = tensor_diags.to(device)
                            vector_X, projectionX, vector_Y, projectionY = model(tensor_day, tensor_diags)

                            if val_step >= total_val_steps:
                                new_batch_size = projectionX.size()[0]
                                criterion = ContrastiveLoss(batch_size=new_batch_size, device=device, temperature=temperature)
                                                
                            loss = criterion(projectionX.type(torch.float32), projectionY.type(torch.float32))

                            valid_running_loss += loss.item()
                            val_step += 1
                            

                epoch_average_val_loss = valid_running_loss / len(valid_loader)

                train_running_loss = 0.0
                valid_running_loss = 0.0

                print(f'Train loss {epoch_average_train_loss}, Validation loss {epoch_average_val_loss}')

                wandb.log({'epoch_average_train_loss':epoch_average_train_loss, 'epoch_average_val_loss':epoch_average_val_loss, 'epoch':epoch+1})
                
                # checkpoint
                if best_valid_loss > epoch_average_val_loss and save_model:
                    print(f'Validation loss decreased {best_valid_loss}==>{epoch_average_val_loss}')
                    best_valid_loss = epoch_average_val_loss
                    save_checkpoint(file_path + '/model.pt', model, optimizer, best_valid_loss)
                    wandb.save(file_path + '/model.pt')
                    stop_training = 0
                else:
                    stop_training +=1
                
                if stop_training == epoch_patience:
                    break

        print('Finished training!')

elif cont_model:
    max_length = {'demographics':5, 'lab_tests':400, 'vitals':200, 'medications':255}
    class MyDataset(Dataset):

        def __init__(self, df, tokenizer, max_length, max_days, pred_window):
            self.df = df
            self.tokenizer = tokenizer
            self.max_days = max_days
            self.pred_window = pred_window
            self.max_len_demo = max_length['demographics']
            self.max_len_labs = max_length['lab_tests']
            self.max_len_vitals = max_length['vitals']
            self.max_len_meds = max_length['medications']
            
        def __len__(self):
            return self.df.shape[0]

        def __getitem__(self, idx):

            return self.make_matrices(idx, self.max_days)
        
        def tokenize(self, text, max_length):
            
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

        def make_matrices(self, idx, max_days):
            info_demo = self.df.demographics_in_visit.values[idx][0]
            info_med = self.df.medications_in_visit.values[idx]
            info_vitals = self.df.vitals_in_visit.values[idx]
            info_labs = self.df.lab_tests_in_visit.values[idx]
            aki_status = self.df.aki_status_in_visit.values[idx]
            days = list(self.df.days.values[idx])

            aki_happened = False
            labels = []
            info_med_list = []
            info_vitals_list = []
            info_labs_list = []

            for day in range(days[0], days[0]+ max_days - self.pred_window):
                if day not in days:
                    labels.append(0)
                    info_med_list.append(self.tokenize('', self.max_len_meds))
                    info_vitals_list.append(self.tokenize('', self.max_len_vitals))
                    info_labs_list.append(self.tokenize('', self.max_len_labs))
                else:
                    i = days.index(day)
                    
                    if (day + self.pred_window) not in days:
                        labels.append(0)
                    else:              
                        if np.isnan(aki_status[i + self.pred_window]):
                            labels.append(0)
                        else:
                            labels.append(aki_status[i + self.pred_window])

                    if str(info_med[i]) == 'nan':
                        info_med_list.append(self.tokenize('[PAD]', self.max_len_meds))
                    else:
                        info_med_list.append(self.tokenize(info_med[i], self.max_len_meds))

                    if str(info_vitals[i]) == 'nan':
                        info_vitals_list.append(self.tokenize('[PAD]', self.max_len_vitals))
                    else:
                        info_vitals_list.append(self.tokenize(info_vitals[i], self.max_len_vitals))

                    if str(info_labs[i]) == 'nan':
                        info_labs_list.append(self.tokenize('[PAD]', self.max_len_labs))
                    else:
                        info_labs_list.append(self.tokenize(info_labs[i], self.max_len_labs))
                        
            info_demo = self.tokenize(info_demo,  self.max_len_demo)

            #make tensors
            tensor_demo = torch.tensor(info_demo, dtype=torch.int32)
            tensor_med = torch.tensor(info_med_list, dtype=torch.int32)
            tensor_vitals = torch.tensor(info_vitals_list, dtype=torch.int32)
            tensor_labs = torch.tensor(info_labs_list, dtype=torch.int32)
            tensor_labels = torch.tensor(labels, dtype=torch.int32)
            
            return (tensor_demo, tensor_med, tensor_vitals, tensor_labs), tensor_labels


    class EHR_model(nn.Module):
        def __init__(self, embedding_size, vocab_size, max_length, pred_window, max_day, drop=0.1, H=128):
            super(EHR_model, self).__init__()
            self.embedding_size = embedding_size
            self.vocab_size = vocab_size
            self.H = H
            self.pred_window = pred_window
            self.max_day = max_day
            self.max_length = max_length

            self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)

            self.fc_med = nn.Linear(max_length['medications'] * 2 * self.H, 256)  #65,280
            self.fc_vit = nn.Linear(max_length['vitals'] * 2 * self.H, 256)   #51,200
            self.fc_lab = nn.Linear(max_length['lab_tests'] * 2 * self.H, 512) #102,400

            self.lstm_day = nn.LSTM(input_size=embedding_size,
                                hidden_size=self.H,
                                num_layers=1,
                                batch_first=True,
                                bidirectional=True)

            self.fc_1 = nn.Linear((self.max_day - self.pred_window) * (256 + 256 + 512) + max_length['demographics']*2*H, 2048)

            self.projection = nn.Sequential(
                nn.ReLU(),
                nn.Linear(in_features=2048, out_features=256)
            )
            self.drop = nn.Dropout(p=drop)
            
        def forward(self, tensor_demo, tensor_med, tensor_vitals, tensor_labs):   
            batch_size = tensor_med.size()[0]
            days = self.max_day

            out_emb_med_demo = self.embedding(tensor_demo.squeeze(1))
            output_lstm_day_demo, _ = self.lstm_day(out_emb_med_demo)
            full_output = output_lstm_day_demo.reshape(batch_size, self.max_length['demographics']* 2 * self.H)

            for d in range(days - self.pred_window):
                # embedding layer applied to all tensors
                out_emb_med = self.embedding(tensor_med[:, d, :].squeeze(1))
                out_emb_vitals = self.embedding(tensor_vitals[:, d, :].squeeze(1))
                out_emb_labs =  self.embedding(tensor_labs[:, d, :].squeeze(1))
                # lstm layer applied to embedded tensors
                output_lstm_day_med = self.fc_med(\
                                        self.lstm_day(out_emb_med)[0]\
                                            .reshape(batch_size, max_length['medications'] * 2 * self.H))

                output_lstm_day_vitals = self.fc_vit(\
                                            self.lstm_day(out_emb_vitals)[0]\
                                                .reshape(batch_size,  max_length['vitals'] * 2 * self.H))

                output_lstm_day_labs = self.fc_lab(\
                                        self.lstm_day(out_emb_labs)[0]\
                                            .reshape(batch_size, max_length['lab_tests']* 2 * self.H))
                                            
                # concatenate for all days
                full_output = torch.cat((full_output, \
                                            output_lstm_day_med,\
                                                output_lstm_day_vitals,\
                                                    output_lstm_day_labs), dim=1)
            
            # print('full_output size: ', full_output.size())
            output_vector = self.fc_1(full_output)

            # the fisrt transformation
            output_vector_X = self.drop(output_vector)
            projection_X = self.projection(output_vector_X)
            # the second transformation
            output_vector_Y = self.drop(output_vector)
            projection_Y = self.projection(output_vector_Y)

            return output_vector_X, projection_X, output_vector_Y, projection_Y



    # TRAIN FUNCTION
    def train(model,
            optimizer,
            train_loader,
            valid_loader,
            batch_size,
            file_path,
            embedding_size,
            device='cpu',
            num_epochs=2,
            epoch_patience=10,
            best_valid_loss = float("Inf"),
            dimension=128,
            save_model=True,
            temperature=0.1):

        
        train_running_loss = 0.0
        valid_running_loss = 0.0
        train_loss_list = []
        valid_loss_list = []
        total_train_steps = len(train_loader)
        total_val_steps = len(valid_loader)
        stop_training = 0


        for epoch in range(num_epochs):
                model.train()
                train_step = 1
                print(f'Epoch {epoch+1}/{num_epochs} training..')
                criterion = ContrastiveLoss(batch_size=batch_size, device=device, temperature=temperature)

                for  (tensor_demo, tensor_med, tensor_vitals, tensor_labs), tensor_labels in train_loader:
                    if train_step % 100==0:
                        print(f'Step {train_step}/{total_train_steps}')

                    d = 0
                    
                
                    vector_X, projectionX, vector_Y, projectionY = model(tensor_demo.to(device),
                                                                    tensor_med.to(device),
                                                                    tensor_vitals.to(device),
                                                                    tensor_labs.to(device))
                                            
                    if train_step >= total_train_steps:
                            new_batch_size = projectionX.size()[0]
                            criterion = ContrastiveLoss(batch_size=new_batch_size, device=device, temperature=temperature)
                    
                    loss = criterion(projectionX.type(torch.float32), projectionY.type(torch.float32))
                    #   print(loss.item())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_running_loss += loss.item()
                    train_step += 1

                    wandb.log({'step_train_loss': loss.item(), 'global_step': train_step})

                epoch_average_train_loss = train_running_loss / len(train_loader)  

                model.eval()
                val_step = 1
                print(f"Validation started..")
                criterion = ContrastiveLoss(batch_size=batch_size, device=device, temperature=temperature)
                with torch.no_grad():
                    for X, Y in valid_loader:
                            vector_X, projectionX, vector_Y, projectionY = model(tensor_demo.to(device),
                                                                    tensor_med.to(device),
                                                                    tensor_vitals.to(device),
                                                                    tensor_labs.to(device))

                            if val_step >= total_val_steps:
                                new_batch_size = projectionX.size()[0]
                                criterion = ContrastiveLoss(batch_size=new_batch_size, device=device, temperature=temperature)
                                                
                            loss = criterion(projectionX.type(torch.float32), projectionY.type(torch.float32))

                            valid_running_loss += loss.item()
                            val_step += 1
                            

                epoch_average_val_loss = valid_running_loss / len(valid_loader)

                train_running_loss = 0.0
                valid_running_loss = 0.0

                print(f'Train loss {epoch_average_train_loss}, Validation loss {epoch_average_val_loss}')

                wandb.log({'epoch_average_train_loss':epoch_average_train_loss, 'epoch_average_val_loss':epoch_average_val_loss, 'epoch':epoch+1})
                
                # checkpoint
                if best_valid_loss > epoch_average_val_loss and save_model:
                    print(f'Validation loss decreased {best_valid_loss}==>{epoch_average_val_loss}')
                    best_valid_loss = epoch_average_val_loss
                    save_checkpoint(file_path + '/model.pt', model, optimizer, best_valid_loss)
                    wandb.save(file_path + '/model.pt')
                    stop_training = 0
                else:
                    stop_training +=1
                
                if stop_training == epoch_patience:
                    break

        print('Finished training!')

elif new_fixed_model:

    # max_length = {'demographics':5, 'diagnoses':35, 'lab_tests':300, 'vitals':31, 'medications':256}
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
            
            # aki_status = self.df.aki_status_in_visit.values[idx]
            days = self.df.days.values[idx]
            # print(idx)

            lab_info_list = []
            med_info_list = []
            vitals_info_list = []
            label = None

            for day in range(days[0], days[0] + self.observing_window + self.pred_window):
                # print('day', day)
                if day not in days:
                    vitals_info_list.append(self.tokenize('', self.max_length['vitals']))
                    lab_info_list.append(self.tokenize('', self.max_length['lab_tests']))
                    med_info_list.append(self.tokenize('', self.max_length['medications']))

                else:
                    i = days.index(day)
                    
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

            #make tensors
            tensor_demo = torch.tensor(demo_info, dtype=torch.int64)
            tensor_diags = torch.tensor(diagnoses_info, dtype=torch.int64)
            tensor_vitals = torch.tensor(vitals_info_list, dtype=torch.int64)
            tensor_labs = torch.tensor(lab_info_list, dtype=torch.int64)
            tensor_meds = torch.tensor(med_info_list, dtype=torch.int64)
            # tensor_labels = torch.tensor(label, dtype=torch.float64)
        
            return tensor_demo, tensor_diags, tensor_vitals, tensor_labs, tensor_meds, hadm_id

    
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
            # print(f'out_emb_diags: ', out_emb_diags.size())
            # print(f'out_emb_demo: ', out_emb_demo.size())
            # lstm for demographic and diagnoses
            out_lstm_diags, _ = self.lstm_day(out_emb_diags)    # [16, 37, 256]
            out_lstm_demo, _ = self.lstm_day(out_emb_demo)      # [16, 7, 256]
            # print(f'out_lstm_diags: ', out_lstm_diags.size())
            # print(f'out_lstm_demo: ', out_lstm_demo.size())
            # reshape and concat demographics and diags
            out_lstm_diags_reshaped = out_lstm_diags.reshape(batch_size, self.max_length_diags * 2 * self.H)
            out_lstm_demo_reshaped = out_lstm_demo.reshape(batch_size, self.max_length_demo * 2 * self.H)
            # print(f'out_lstm_diags_reshaped', out_lstm_diags_reshaped.size())
            # print(f'out_lstm_demo_reshaped', out_lstm_demo_reshaped.size())
            full_output = torch.cat([out_lstm_demo_reshaped, out_lstm_diags_reshaped], dim=1)   # [16, 11264]
            # print(f'full_output', full_output.size())

            for d in range(self.observing_window):
                # embedding layer applied to all tensors 
                out_med_emb = self.embedding(tensor_med[:, d, :].squeeze(1))
                out_vitals_emb = self.embedding(tensor_vitals[:, d, :].squeeze(1))
                out_labs_emb = self.embedding(tensor_labs[:, d, :].squeeze(1))
                # print('out_med_emb', out_med_emb.size())
                # print('out_vitals_emb', out_vitals_emb.size())
                # print('out_labs_emb', out_labs_emb.size())

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

                # print('output_lstm_med', output_lstm_med.size())                   
                # print('output_lstm_vitals', output_lstm_vitals.size())                   
                # print('output_lstm_labs', output_lstm_labs.size())    
                # print('--------------')               
                # concatenate for all * days
                full_output = torch.cat((full_output, \
                                            output_lstm_med,\
                                                output_lstm_vitals,\
                                                    output_lstm_labs), dim=1) # 

            # print('full_output size: ', full_output.size(), '\n')
            output = self.fc_adm(full_output)
            # print('fc_adm: ', output.size(), '\n')
            output_vector, _ = self.lstm_adm(output)
            # print('output_vector: ', output_vector.size(), '\n')

            # the fisrt transformation
            output_vector_X = self.drop(output_vector)
            projection_X = self.projection(output_vector_X)
            # the second transformation
            output_vector_Y = self.drop(output_vector)
            projection_Y = self.projection(output_vector_Y)


            return output_vector_X, projection_X, output_vector_Y, projection_Y
    
    def train(model,
            optimizer,
            train_loader,
            valid_loader,
            batch_size,
            file_path,
            embedding_size,
            device='cpu',
            num_epochs=2,
            epoch_patience=10,
            best_valid_loss = float("Inf"),
            dimension=128,
            save_model=True,
            temperature=0.1):

        
        train_running_loss = 0.0
        valid_running_loss = 0.0
        train_loss_list = []
        valid_loss_list = []
        total_train_steps = len(train_loader)
        total_val_steps = len(valid_loader)
        stop_training = 0


        for epoch in range(num_epochs):
                model.train()
                train_step = 1
                print(f'Epoch {epoch+1}/{num_epochs} training..')
                criterion = ContrastiveLoss(batch_size=batch_size, device=device, temperature=temperature)

                for  tensor_demo, tensor_diags, tensor_vitals, tensor_labs, tensor_meds, idx in train_loader:
                    if train_step % 100==0:
                        print(f'Step {train_step}/{total_train_steps}')
                    # print(f'Step {train_step}/{total_train_steps}')

                    vector_X, projectionX, vector_Y, projectionY = model(tensor_demo.to(device), tensor_diags.to(device),\
                                                                        tensor_meds.to(device), tensor_vitals.to(device),\
                                                                        tensor_labs.to(device))
                                            
                    if train_step >= total_train_steps:
                            new_batch_size = projectionX.size()[0]
                            criterion = ContrastiveLoss(batch_size=new_batch_size, device=device, temperature=temperature)
                    
                    loss = criterion(projectionX.type(torch.float32), projectionY.type(torch.float32))
                    #   print(loss.item())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_running_loss += loss.item()
                    train_step += 1

                    wandb.log({'step_train_loss': loss.item(), 'global_step': train_step})

                epoch_average_train_loss = train_running_loss / len(train_loader)  

                model.eval()
                val_step = 1
                print(f"Validation started..")
                criterion = ContrastiveLoss(batch_size=batch_size, device=device, temperature=temperature)
                with torch.no_grad():
                    for tensor_demo, tensor_diags, tensor_vitals, tensor_labs, tensor_meds, idx in valid_loader:
                            vector_X, projectionX, vector_Y, projectionY = model(tensor_demo.to(device), tensor_diags.to(device),\
                                                                        tensor_meds.to(device), tensor_vitals.to(device),\
                                                                        tensor_labs.to(device))

                            if val_step >= total_val_steps:
                                new_batch_size = projectionX.size()[0]
                                criterion = ContrastiveLoss(batch_size=new_batch_size, device=device, temperature=temperature)
                                                
                            loss = criterion(projectionX.type(torch.float32), projectionY.type(torch.float32))

                            valid_running_loss += loss.item()
                            val_step += 1
                            

                epoch_average_val_loss = valid_running_loss / len(valid_loader)

                train_running_loss = 0.0
                valid_running_loss = 0.0

                print(f'Train loss {epoch_average_train_loss}, Validation loss {epoch_average_val_loss}')

                wandb.log({'epoch_average_train_loss':epoch_average_train_loss, 'epoch_average_val_loss':epoch_average_val_loss, 'epoch':epoch+1})
                
                # checkpoint
                if best_valid_loss > epoch_average_val_loss and save_model:
                    print(f'Validation loss decreased {best_valid_loss}==>{epoch_average_val_loss}')
                    best_valid_loss = epoch_average_val_loss
                    save_checkpoint(file_path + '/model.pt', model, optimizer, best_valid_loss)
                    wandb.save(file_path + '/model.pt')
                    stop_training = 0
                else:
                    stop_training +=1
                
                if stop_training == epoch_patience:
                    break

        print('Finished training!')

elif model_three_stages:

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
            # print(hadm_id)
            # print(days)
            day_info_list = []

            for day in range(0, self.observing_window + self.pred_window):
                
                if day not in days:
                    day_info_list.append(self.tokenize('', self.max_length_day))
                else:
                    # print('day', day)
                    i = days.index(day)

                    if (str(day_info[i]) == 'nan') or (day_info[i] == np.nan):
                        day_info_list.append(self.tokenize('PAD', self.max_length_day))
                    else:
                        day_info_list.append(self.tokenize(day_info[i], self.max_length_day))

            # diagnoses
            if (str(diagnoses_info) == 'nan') or (diagnoses_info == np.nan):
                diagnoses_info = self.tokenize('PAD', self.max_length_diags)
            else:
                diagnoses_info = self.tokenize(diagnoses_info, self.max_length_diags)


            #make tensors
            tensor_day = torch.tensor(day_info_list[:self.observing_window], dtype=torch.int64)
            tensor_diags = torch.tensor(diagnoses_info, dtype=torch.int64)
            tensor_labels = torch.tensor([], dtype=torch.int64)

            return tensor_day, tensor_diags, tensor_labels, hadm_id

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

    def train(model,
            optimizer,
            train_loader,
            valid_loader,
            batch_size,
            file_path,
            embedding_size,
            device='cpu',
            num_epochs=2,
            epoch_patience=10,
            best_valid_loss = float("Inf"),
            dimension=128,
            save_model=True,
            temperature=0.1):

        
        train_running_loss = 0.0
        valid_running_loss = 0.0
        train_loss_list = []
        valid_loss_list = []
        total_train_steps = len(train_loader)
        total_val_steps = len(valid_loader)
        stop_training = 0


        for epoch in range(num_epochs):
                model.train()
                train_step = 1
                print(f'Epoch {epoch+1}/{num_epochs} training..')
                criterion = ContrastiveLoss(batch_size=batch_size, device=device, temperature=temperature)

                for  tensor_day, tensor_diags, tensor_labels, hadm_id in train_loader:
                    # transferring everything to GPU
                    tensor_day = tensor_day.to(device)
                    tensor_diags = tensor_diags.to(device)

                    if train_step % 100==0:
                        print(f'Step {train_step}/{total_train_steps}')
                    # print(f'Step {train_step}/{total_train_steps}')

                    vector_X, projectionX, vector_Y, projectionY = model(tensor_day, tensor_diags)
                                            
                    if train_step >= total_train_steps:
                            new_batch_size = projectionX.size()[0]
                            criterion = ContrastiveLoss(batch_size=new_batch_size, device=device, temperature=temperature)
                    
                    loss = criterion(projectionX.type(torch.float32), projectionY.type(torch.float32))
                    #   print(loss.item())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_running_loss += loss.item()
                    train_step += 1

                    wandb.log({'step_train_loss': loss.item(), 'global_step': train_step})

                epoch_average_train_loss = train_running_loss / len(train_loader)  

                model.eval()
                val_step = 1
                print(f"Validation started..")
                criterion = ContrastiveLoss(batch_size=batch_size, device=device, temperature=temperature)
                with torch.no_grad():
                    for tensor_day, tensor_diags, tensor_labels, hadm_id in valid_loader:
                            tensor_day = tensor_day.to(device)
                            tensor_diags = tensor_diags.to(device)
                            vector_X, projectionX, vector_Y, projectionY = model(tensor_day, tensor_diags)

                            if val_step >= total_val_steps:
                                new_batch_size = projectionX.size()[0]
                                criterion = ContrastiveLoss(batch_size=new_batch_size, device=device, temperature=temperature)
                                                
                            loss = criterion(projectionX.type(torch.float32), projectionY.type(torch.float32))

                            valid_running_loss += loss.item()
                            val_step += 1
                            

                epoch_average_val_loss = valid_running_loss / len(valid_loader)

                train_running_loss = 0.0
                valid_running_loss = 0.0

                print(f'Train loss {epoch_average_train_loss}, Validation loss {epoch_average_val_loss}')

                wandb.log({'epoch_average_train_loss':epoch_average_train_loss, 'epoch_average_val_loss':epoch_average_val_loss, 'epoch':epoch+1})
                
                # checkpoint
                if best_valid_loss > epoch_average_val_loss and save_model:
                    print(f'Validation loss decreased {best_valid_loss}==>{epoch_average_val_loss}')
                    best_valid_loss = epoch_average_val_loss
                    save_checkpoint(file_path + '/model.pt', model, optimizer, best_valid_loss)
                    wandb.save(file_path + '/model.pt')
                    stop_training = 0
                else:
                    stop_training +=1
                
                if stop_training == epoch_patience:
                    break

        print('Finished training!')



def main(project_name,  num_epochs, pred_window, max_day, experiment, additional_name='', PRETRAINED_PATH=None, drop=0.1, \
    temperature=0.5, embedding_size=200, min_frequency=10, BATCH_SIZE=16, small_dataset=True, \
        LR=0.0001, save_model=False, use_gpu=True, saving_folder_name=None, wandb_mode = 'online', \
            run_id=None, diagnoses='titles'):
    
    if use_gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device='cpu'
    print('device: ', device)
    
    #paths
    CURR_PATH = os.getcwd() + '/LSTM/'
    PKL_PATH = CURR_PATH+'/pickles/'
    DF_PATH = CURR_PATH +'/dataframes/'
    # destination_folder = '/l/users/svetlana.maslenkova/models' + '/pretraining/three_stages/'
    destination_folder = '/home/svetlanamaslenkova/Documents/AKI_deep/pretraining/'

    if diagnoses=='icd':
        TOKENIZER_PATH = CURR_PATH + '/aki_prediction' + '/tokenizer.json'
        TXT_DIR_TRAIN = CURR_PATH + '/aki_prediction' + '/txt_files/train'
    elif diagnoses=='titles':
        TOKENIZER_PATH = CURR_PATH + '/aki_prediction'+ '/tokenizer_titles.json'
        TXT_DIR_TRAIN = CURR_PATH + '/aki_prediction'+ '/txt_files/titles_diags'

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
    # max_length = {'demographics':5, 'lab_tests':400, 'vitals':200, 'medications':255}
    max_length = {'demographics':5+2, 'diagnoses':35+2, 'lab_tests':300+2, 'vitals':31+2, 'medications':256+2}
    vocab_size = tokenizer.get_vocab_size()
    print(f'Vocab size: {vocab_size}')
    embedding_size = 200
    dimension = 128

    with open(DF_PATH + 'pid_train_df_pretraining.pkl', 'rb') as f:
        pid_train_df = pickle.load(f)

    with open(DF_PATH + 'pid_val_df_pretraining.pkl', 'rb') as f:
        pid_val_df = pickle.load(f)

    if small_dataset: frac=0.001
    else: frac=1
    
    # pid_train_df_small = pid_train_df.sample(frac=frac)
    # pid_val_df_small = pid_val_df.sample(frac=frac)

    if fixed_model_with_diags:
        train_dataset = MyDataset(pid_train_df.sample(frac=frac), tokenizer=tokenizer, max_length_day=400)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        val_dataset = MyDataset(pid_val_df.sample(frac=frac), tokenizer=tokenizer, max_length_day=400)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    elif cont_model:
        train_dataset = MyDataset(pid_train_df_small, tokenizer=tokenizer, max_length=max_length)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        val_dataset = MyDataset(pid_val_df_small, tokenizer=tokenizer, max_length=max_length)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    elif new_fixed_model:
        train_dataset = MyDataset(pid_train_df.sample(frac=frac), tokenizer=tokenizer, max_length=max_length)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        val_dataset = MyDataset(pid_val_df.sample(frac=frac), tokenizer=tokenizer, max_length=max_length)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    elif model_three_stages:
        train_dataset = MyDataset(pid_train_df.sample(frac=frac), tokenizer=tokenizer, diags=diagnoses, max_length_day=400)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        val_dataset = MyDataset(pid_val_df.sample(frac=frac), tokenizer=tokenizer, diags=diagnoses, max_length_day=400)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print('DATA SHAPES: ')
    print('train data shape: ', len(train_loader)*BATCH_SIZE)
    print('val data shape: ', len(val_loader)*BATCH_SIZE)

    if fixed_model_with_diags:
        model = EHR_PRETRAINING(max_length=400, vocab_size=vocab_size, device=device).to(device)
    elif cont_model:
        model = EHR_model(embedding_size=embedding_size, vocab_size=vocab_size, max_length=max_length, pred_window=pred_window, max_day=max_day, drop=0.1).to(device)
    elif new_fixed_model:
        model = EHR_PRETRAINING(max_length=max_length, vocab_size=vocab_size, device=device, pred_window=2, observing_window=3,  H=128, embedding_size=200, drop=0.6).to(device)
    elif  model_three_stages:
        max_length=400
        model = EHR_PRETRAINING(max_length, vocab_size, device, diags=diagnoses, pred_window=2, observing_window=2,  H=dimension, embedding_size=embedding_size, drop=drop).to(device)

    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    # Decay LR by a factor of 0.1 every 7 epochs
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    if PRETRAINED_PATH is not None:
        load_checkpoint(PRETRAINED_PATH, model, optimizer, device=device)
    
    train_params = {'model':model,
                    'optimizer':optimizer,
                    'train_loader':train_loader,
                    'valid_loader':val_loader,
                    'batch_size':BATCH_SIZE,
                    'embedding_size':embedding_size,
                    'file_path':destination_folder,
                    'num_epochs':num_epochs,
                    'device':device,
                    'save_model':save_model,
                    'temperature':temperature
    }

    num_samples = len(train_loader)*BATCH_SIZE // 1000

    if saving_folder_name is None:
        saving_folder_name = additional_name + 'STG' + '_bs' + str(BATCH_SIZE) +'_' + str(num_samples) + 'k_' + diagnoses + '_lr'+ str(LR) + '_Adam' + '_temp' + str(temperature) + '_drop' + str(drop)
    file_path = destination_folder + saving_folder_name
    train_params['file_path'] = file_path

    print(f'\n\nMODEL PATH: {file_path}')
    if not os.path.exists(file_path):
        os.makedirs(file_path)
        
    run_name = saving_folder_name
    print('Run name: ', run_name)
    args = {'optimizer':'Adam', 'LR':LR, 'min_frequency':min_frequency, 'dropout':drop, \
        'vocab_size':vocab_size, 'embedding_size':embedding_size, 'pretrained':'lstm_adm', \
            'temperature':temperature, 'batch_size':BATCH_SIZE,  'experiment':experiment}

    if run_id is None:    
        run_id = wandb.util.generate_id()  
        resume = 'allow' 
    else:
        resume = 'must'

    print('Run id is: ', run_id)
    wandb.init(project=project_name, name=run_name, mode=wandb_mode, config=args, id=run_id, resume=resume)
    train(**train_params)
    wandb.finish()




# main(project_name='Contrastive-loss-pretraining', saving_folder_name=None, num_epochs=20, embedding_size=200, max_length=max_length, pred_window=1, max_day=7, min_frequency=1, BATCH_SIZE=128, small_dataset=False, LR=0.00001, save_model=True, use_gpu=True, log_results=True)

#  28722
# main(project_name='Contrastive-loss-pretraining', saving_folder_name=None, num_epochs=20, temperature=0.05, embedding_size=200, max_length=max_length, pred_window=1, max_day=7, min_frequency=1, BATCH_SIZE=128, small_dataset=False, LR=0.00001, save_model=True, use_gpu=True, log_results=True)

# 28723
# main(project_name='Contrastive-loss-pretraining', saving_folder_name=None, num_epochs=20, temperature=0.01, embedding_size=200, max_length=max_length, pred_window=1, max_day=7, min_frequency=1, BATCH_SIZE=128, small_dataset=False, LR=0.00001, save_model=True, use_gpu=True, log_results=True)

# 28724
# main(project_name='Contrastive-loss-pretraining', saving_folder_name=None, num_epochs=20, temperature=0.1, embedding_size=200, max_length=max_length, pred_window=1, max_day=7, min_frequency=1, BATCH_SIZE=128, small_dataset=False, LR=0.00001, save_model=True, use_gpu=True, log_results=True)


# #  29402,  29763, 29800 (continue training)
# PRETRAINED_PATH = '/home/svetlana.maslenkova/LSTM/pretraining/fc1/CL_FC1_bs128_142k_lr1e-05_Adam_temp0.05/model.pt'
# main(project_name='Contrastive-loss-pretraining', saving_folder_name='CL_FC1_bs128_142k_lr1e-05_Adam_temp0.05', num_epochs=27, temperature=0.05, embedding_size=200, max_length=max_length, pred_window=1, max_day=7, min_frequency=1, BATCH_SIZE=512, small_dataset=False, LR=0.00001, save_model=True, use_gpu=True, PRETRAINED_PATH=PRETRAINED_PATH, resume_run='2hrgwbg0')

#############################################################################
     ######## Fixed model with diags pretraining experiments ###########
#############################################################################  

# # test run
# main(project_name='test', num_epochs=1, pred_window=None, max_day=None, PRETRAINED_PATH=None, drop=0.1, \
#     temperature=0.1, embedding_size=200, min_frequency=5, BATCH_SIZE=16, small_dataset=True, \
#         LR=0.00001, save_model=True, use_gpu=False, saving_folder_name='test', wandb_mode = 'online', \
#             run_id=None)


# # 35413, 35420(with inner drop): drop=0.1, t=0.1
# main(project_name='Contrastive-loss-pretraining', num_epochs=20, pred_window=None, max_day=None, PRETRAINED_PATH=None, drop=0.1, \
#     temperature=0.1, embedding_size=200, min_frequency=5, BATCH_SIZE=80, small_dataset=False, \
#         LR=0.00001, save_model=True, use_gpu=True, saving_folder_name=None, wandb_mode = 'online', \
#             run_id=None)

# # 35421: drop=0.1, t=0.01
# main(project_name='Contrastive-loss-pretraining', num_epochs=20, pred_window=None, max_day=None, PRETRAINED_PATH=None, drop=0.1, \
#     temperature=0.1, embedding_size=200, min_frequency=5, BATCH_SIZE=64, small_dataset=False, \
#         LR=0.00001, save_model=True, use_gpu=True, saving_folder_name=None, wandb_mode = 'online', \
#             run_id=None)

# # 35422: drop=0.1, t=0.05
# main(project_name='Contrastive-loss-pretraining', num_epochs=20, pred_window=None, max_day=None, PRETRAINED_PATH=None, drop=0.1, \
#     temperature=0.05, embedding_size=200, min_frequency=5, BATCH_SIZE=64, small_dataset=False, \
#         LR=0.00001, save_model=True, use_gpu=True, saving_folder_name=None, wandb_mode = 'online', \
#             run_id=None)

# # # 35423: drop=0.2, t=0.1
# main(project_name='Contrastive-loss-pretraining', num_epochs=20, pred_window=None, max_day=None, PRETRAINED_PATH=None, drop=0.2, \
#     temperature=0.1, embedding_size=200, min_frequency=5, BATCH_SIZE=64, small_dataset=False, \
#         LR=0.00001, save_model=True, use_gpu=True, saving_folder_name=None, wandb_mode = 'online', \
#             run_id=None)

# # # 35426: drop=0.2, t=0.01
# main(project_name='Contrastive-loss-pretraining', num_epochs=20, pred_window=None, max_day=None, PRETRAINED_PATH=None, drop=0.2, \
#     temperature=0.01, embedding_size=200, min_frequency=5, BATCH_SIZE=64, small_dataset=False, \
#         LR=0.00001, save_model=True, use_gpu=True, saving_folder_name=None, wandb_mode = 'online', \
#             run_id=None)

# # # 35427: drop=0.2, t=0.05
# main(project_name='Contrastive-loss-pretraining', num_epochs=20, pred_window=None, max_day=None, PRETRAINED_PATH=None, drop=0.2, \
#     temperature=0.05, embedding_size=200, min_frequency=5, BATCH_SIZE=64, small_dataset=False, \
#         LR=0.00001, save_model=True, use_gpu=True, saving_folder_name=None, wandb_mode = 'online', \
#             run_id=None)


#####################################################################################################################
        ######## Fixed model with diags pretraining experiments NEW DATASET CLASS, NEW TOKENIZER ###########
#####################################################################################################################

# # # test run
# main(project_name='test', num_epochs=1, pred_window=None, max_day=None, PRETRAINED_PATH=None, drop=0.1, \
#     temperature=0.1, embedding_size=200, min_frequency=5, BATCH_SIZE=128, small_dataset=True, \
#         LR=0.00001, save_model=True, use_gpu=False, saving_folder_name='test', wandb_mode = 'disabled', \
#             run_id=None)

# # 39072: bs 512
# main(project_name='Contrastive-loss-pretraining', num_epochs=50, pred_window=None, max_day=None, PRETRAINED_PATH=None, drop=0.1, \
#     temperature=0.1, embedding_size=200, min_frequency=10, BATCH_SIZE=512, small_dataset=False, \
#         LR=0.00001, save_model=True, use_gpu=True, saving_folder_name=None, wandb_mode = 'online', \
#             run_id=None)

# # 39076: bs 800
# main(project_name='Contrastive-loss-pretraining', num_epochs=50, pred_window=None, max_day=None, PRETRAINED_PATH=None, drop=0.1, \
#     temperature=0.1, embedding_size=200, min_frequency=10, BATCH_SIZE=800, small_dataset=False, \
#         LR=0.00001, save_model=True, use_gpu=True, saving_folder_name=None, wandb_mode = 'online', \
#             run_id=None)


#####################################################################################################################
                                    ######## Three stages model ###########
#####################################################################################################################

# main(project_name='test',  num_epochs=2, pred_window=2, max_day=None, experiment='STG', additional_name='', \
#     PRETRAINED_PATH=None, drop=0.1, temperature=0.5, embedding_size=200, min_frequency=10, BATCH_SIZE=128, \
#     small_dataset=True, LR=0.0001, save_model=False, use_gpu=False, saving_folder_name='test_model', wandb_mode = 'disabled', \
#             run_id=None, diagnoses='icd')

# # 48666, 48668 bad vocab
# main(project_name='Contrastive-loss-pretraining',  num_epochs=100, pred_window=2, max_day=None, additional_name='', \
#     PRETRAINED_PATH=None, drop=0.1, temperature=0.5, embedding_size=200, min_frequency=1, BATCH_SIZE=300, \
#     small_dataset=False, LR=0.0001, save_model=True, use_gpu=True, saving_folder_name=None, wandb_mode = 'online', \
#             run_id=None, diagnoses='titles')

# # # 48946 bs8, 48947 bs32, 48952 bs256, 48955 bs521
# main(project_name='Contrastive-loss-pretraining',  num_epochs=100, pred_window=2, max_day=None, additional_name='', \
#     PRETRAINED_PATH=None, drop=0.1, temperature=0.1, embedding_size=200, min_frequency=1, BATCH_SIZE=521, \
#     small_dataset=False, LR=0.0001, save_model=True, use_gpu=True, saving_folder_name=None, wandb_mode = 'online', \
#             run_id=None, diagnoses='titles')

# # 48961 bs512, 48962 bs256
# main(project_name='Contrastive-loss-pretraining',  num_epochs=100, pred_window=2, max_day=None, experiment='STG', additional_name='', \
#     PRETRAINED_PATH=None, drop=0.1, temperature=0.05, embedding_size=200, min_frequency=10, BATCH_SIZE=256, \
#     small_dataset=False, LR=0.0001, save_model=True, use_gpu=True, saving_folder_name=None, wandb_mode = 'online', \
#             run_id=None, diagnoses='titles')

# 48980 bs512, 48983 bs256
# main(project_name='Contrastive-loss-pretraining',  num_epochs=100, pred_window=2, max_day=None, experiment='STG', additional_name='', \
#     PRETRAINED_PATH=None, drop=0.1, temperature=0.1, embedding_size=200, min_frequency=10, BATCH_SIZE=256, \
#     small_dataset=False, LR=0.0001, save_model=True, use_gpu=True, saving_folder_name=None, wandb_mode = 'online', \
#             run_id=None, diagnoses='icd')

# 48984 bs512, 48985 bs256
# main(project_name='Contrastive-loss-pretraining',  num_epochs=100, pred_window=2, max_day=None, experiment='STG', additional_name='', \
#     PRETRAINED_PATH=None, drop=0.1, temperature=0.05, embedding_size=200, min_frequency=10, BATCH_SIZE=256, \
#     small_dataset=False, LR=0.0001, save_model=True, use_gpu=True, saving_folder_name=None, wandb_mode = 'online', \
#             run_id=None, diagnoses='icd')

main(project_name='test',  num_epochs=1, pred_window=2, max_day=None, experiment='STG', additional_name='', \
    PRETRAINED_PATH=None, drop=0.1, temperature=0.05, embedding_size=200, min_frequency=10, BATCH_SIZE=128, \
    small_dataset=True, LR=0.0001, save_model=True, use_gpu=False, saving_folder_name='test_model', wandb_mode = 'disabled', \
            run_id=None, diagnoses='icd')