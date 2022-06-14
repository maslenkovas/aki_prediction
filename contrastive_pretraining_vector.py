# IMPORTS
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tokenizers import  Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import glob
from os.path import exists
import os

import pickle5 as pickle
import wandb

import pandas as pd
import numpy as np




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


def load_checkpoint(load_path, model, optimizer):

    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    
    return state_dict['valid_loss']


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
          epoch_patience=5,
          best_valid_loss = float("Inf"),
          dimension=128,
          save_model=True,
          log_results=False):

      
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
            criterion = ContrastiveLoss(batch_size=batch_size, device=device)

            for  (tensor_demo, tensor_med, tensor_vitals, tensor_labs), tensor_labels in train_loader:
                  print(f'Step {train_step}/{total_train_steps}')
                  d = 0
                  
            
                  vector_X, projectionX, vector_Y, projectionY = model(tensor_demo.to(device),
                                                                tensor_med.to(device),
                                                                tensor_vitals.to(device),
                                                                tensor_labs.to(device))
                                          
                  if train_step >= total_train_steps:
                        new_batch_size = projectionX.size()[0]
                        criterion = ContrastiveLoss(batch_size=new_batch_size, device=device)
                  
                  loss = criterion(projectionX.type(torch.float32), projectionY.type(torch.float32))
                #   print(loss.item())
                  optimizer.zero_grad()
                  loss.backward()
                  optimizer.step()

                  train_running_loss += loss.item()
                  train_step += 1
                  if log_results:
                        wandb.log({'step_train_loss': loss.item(), 'global_step': train_step})

            epoch_average_train_loss = train_running_loss / len(train_loader)  

            model.eval()
            val_step = 1
            print(f"Validation started..")
            criterion = ContrastiveLoss(batch_size=batch_size, device=device)
            with torch.no_grad():
                  for X, Y in valid_loader:
                        vector_X, projectionX, vector_Y, projectionY = model(tensor_demo.to(device),
                                                                tensor_med.to(device),
                                                                tensor_vitals.to(device),
                                                                tensor_labs.to(device))

                        if val_step >= total_val_steps:
                              new_batch_size = projectionX.size()[0]
                              criterion = ContrastiveLoss(batch_size=new_batch_size, device=device)
                                               
                        loss = criterion(projectionX.type(torch.float32), projectionY.type(torch.float32))

                        valid_running_loss += loss.item()
                        val_step += 1
                        

            epoch_average_val_loss = valid_running_loss / len(valid_loader)

            train_running_loss = 0.0
            valid_running_loss = 0.0

            print(f'Train loss {epoch_average_train_loss}, Validation loss {epoch_average_val_loss}')
            if log_results:
                  wandb.log({'epoch_average_train_loss':epoch_average_train_loss, 'epoch_average_val_loss':epoch_average_val_loss, 'epoch':epoch+1})
            
            # checkpoint
            if best_valid_loss > epoch_average_val_loss and save_model:
                  best_valid_loss = epoch_average_val_loss
                  save_checkpoint(file_path + '/model.pt', model, optimizer, best_valid_loss)
            else:
                  stop_training +=1
            
            if stop_training == epoch_patience:
                  break

      print('Finished training!')


def main(project_name, num_epochs, max_length, pred_window, max_day, drop=0.1, embedding_size=200, min_frequency=1, BATCH_SIZE=16, small_dataset=True, LR=0.00001, save_model=False, use_gpu=True, saving_folder_name=None, log_results=False):
    
    if use_gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device='cpu'
    print(device)
    
    #paths
    CURR_PATH = os.getcwd()
    PKL_PATH = CURR_PATH+'/pickles/'
    DF_PATH = CURR_PATH +'/dataframes/'
    TXT_DIR_TRAIN = CURR_PATH + '/txt_files/train'
    destination_folder = CURR_PATH + '/pretraining/fc1/'


    # Training the tokenizer
    if exists(CURR_PATH + '/tokenizer.json'):
        tokenizer = Tokenizer.from_file(CURR_PATH + '/tokenizer.json')
    else:
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(special_tokens=["[PAD]", "[UNK]"], min_frequency=1)
        files = glob.glob(TXT_DIR_TRAIN+'/*')
        tokenizer.train(files, trainer)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # variables for classes
    max_length = {'demographics':5, 'lab_tests':400, 'vitals':200, 'medications':255}
    vocab_size = tokenizer.get_vocab_size()
    print(f'Vocab size: {vocab_size}')

    with open(DF_PATH + 'pid_train_df_pretraining.pkl', 'rb') as f:
        pid_train_df = pickle.load(f)

    with open(DF_PATH + 'pid_val_df_pretraining.pkl', 'rb') as f:
        pid_val_df = pickle.load(f)

    if small_dataset:
        pid_train_df_small = pid_train_df.sample(frac=0.0008)
        pid_val_df_small = pid_val_df.sample(frac=0.015)

        train_dataset = MyDataset(pid_train_df_small, tokenizer=tokenizer, max_length=max_length, max_days=max_day, pred_window=pred_window)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

        val_dataset = MyDataset(pid_val_df_small, tokenizer=tokenizer, max_length=max_length, max_days=max_day, pred_window=pred_window)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    else:
        train_dataset = MyDataset(pid_train_df, tokenizer=tokenizer, max_length=max_length, max_days=max_day, pred_window=pred_window)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

        val_dataset = MyDataset(pid_val_df, tokenizer=tokenizer, max_length=max_length, max_days=max_day, pred_window=pred_window)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    print('DATA SHAPES: ')
    print('train data shape: ', len(train_loader)*BATCH_SIZE)
    print('val data shape: ', len(val_loader)*BATCH_SIZE)

    model = EHR_model(embedding_size=embedding_size, vocab_size=vocab_size, max_length=max_length, pred_window=pred_window, max_day=max_day, drop=0.1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

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
                    'log_results':log_results
                    
    }

    num_samples = (len(train_loader)+len(val_loader))*BATCH_SIZE // 1000

    if saving_folder_name is None:
        saving_folder_name = 'CL_FC1_' + str(num_samples) + 'k' + '_lr'+ str(LR) + '_Adam'
    file_path = destination_folder + saving_folder_name
    train_params['file_path'] = file_path

    print(f'\n\nMODEL PATH: {file_path}')
    if not os.path.exists(file_path):
        os.makedirs(file_path)
        
    run_name = saving_folder_name

    args = {'optimizer':optimizer, 'LR':LR, 'min_frequency':min_frequency, 'dropout':drop, 'vocab_size':vocab_size, 'embedding_size':embedding_size, 'pretrained':'FC1'}

    if log_results:
        wandb.init(project=project_name, name=run_name, mode='online')
        train(**train_params)
        wandb.finish()
    else:
        train(**train_params)



main(project_name='Contrastive-loss-pretraining', saving_folder_name=None, num_epochs=20, embedding_size=200, max_length=max_length, pred_window=1, max_day=7, min_frequency=1, BATCH_SIZE=128, small_dataset=False, LR=0.00001, save_model=True, use_gpu=True, log_results=True)