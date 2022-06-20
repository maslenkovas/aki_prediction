# IMPORTS
from distutils.command.config import config
import pickle5 as pickle

# Libraries
import matplotlib.pyplot as plt
import pandas as pd
import wandb
import os
from os.path import exists

# Evaluation
from sklearn.metrics import f1_score, multilabel_confusion_matrix, accuracy_score, classification_report
import seaborn as sns

# Tokenization
from tokenizers import  Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
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

# DATASET CLASS
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
        days = self.df.days.values[idx]

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
                    if ((i + self.pred_window) >= len(aki_status)) or np.isnan(aki_status[i + self.pred_window]):
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

# MODELS
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



class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


class LSTM_model(nn.Module):

    def __init__(self, pretrained_model, max_length, pred_window, vocab_size, H=128,  max_day=7, embedding_size=200):
        super(LSTM_model, self).__init__()

		# Hyperparameters
        self.max_day = max_day
        self.pred_window = pred_window
        L = (self.max_day+1) * (256 + 256 + 512) + 1280
        self.H = H
        self.max_length = max_length
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        
        self.embedding = pretrained_model.embedding

        self.fc_med = pretrained_model.fc_med
        self.fc_vit = pretrained_model.fc_vit
        self.fc_lab = pretrained_model.fc_lab
        self.lstm_day = pretrained_model.lstm_day
        self.fc_1 = pretrained_model.fc_1

        self.lstm_adm = nn.LSTM(input_size=2048,
                            hidden_size=self.H,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=False)

        self.drop = nn.Dropout(p=0.5)

        self.fc_2 = nn.Linear(self.H, (self.max_day - self.pred_window))
        
        # self.sigmoid = nn.Sigmoid()


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
                                        
            # concatenate for all * days
            full_output = torch.cat((full_output, \
                                        output_lstm_day_med,\
                                            output_lstm_day_vitals,\
                                                output_lstm_day_labs), dim=1)
        
        # print('full_output size: ', full_output.size())
        output = self.fc_1(full_output)
        output, _ = self.lstm_adm(output)
        output = self.drop(output)
        output = self.fc_2(output)
        output = torch.squeeze(output, 1)
        # if self.criterion == 'BCELoss':
        #     output = self.sigmoid(output)

        return output


# FUNCTIONS
def initialize_model(pretrained_path, embedding_size, vocab_size, gpu=False, H=128, max_length=max_length, max_day=7):
    if gpu==True:
      map_location = torch.device('cuda')
    else:
      map_location = torch.device('cpu')

    # initializing the pretrained embeddings model
    pretrained_model = EHR_Embedding(embedding_size=embedding_size, vocab_size=vocab_size)
    pretrained_model.load_state_dict(torch.load(pretrained_path, map_location=map_location)['model_state_dict'])
    num_ftrs = pretrained_model.projection[1].in_features
    pretrained_model.projection = Identity()

    # creating the LSTM model with pretrained embedding layer
    model_ft = LSTM_model(pretrained_model=pretrained_model, H=H, max_length=max_length, max_day=max_day, vocab_size=vocab_size, embedding_size=embedding_size)
    
    return model_ft

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

# Evaluation Function
def get_list_correct_preds(output, target, correct_preds):
    # correct_preds - the list of size (number of samples) where 1 when predicted value 
    # is equal to target, othrwise 0.

    # predicted = (output > threshold).int()
    predicted = output
    for i in range(target.size(0)):
        true = target[i].cpu().detach().numpy()
        pred = predicted[i].cpu().detach().numpy()

        if any(target[i] == 1):
            idx_of_one = np.where(true==1)[0][0]
            if (pred[idx_of_one] == 1) & all(pred[:idx_of_one] == 0):
                correct_preds.append(1)
            else:
                correct_preds.append(0)
        
        else:
            if all(pred == 0):
                correct_preds.append(1)
            else:
                correct_preds.append(0)        
                       
    return correct_preds


def evaluate(model, test_loader, device, threshold=0.5, log_res=True):
    
    stacked_labels = torch.tensor([]).to(device)
    stacked_preds = torch.tensor([]).to(device)
    model.eval()
    step = 1
    correct_preds = []

    with torch.no_grad():
        for (tensor_demo, tensor_med, tensor_vitals, tensor_labs), tensor_labels in test_loader:
            # print(f'Step {step}/{len(test_loader)}' )
            labels = tensor_labels.to(device)
            demo = tensor_demo.to(device)
            med = tensor_med.to(device)
            vitals = tensor_vitals.to(device)
            labs = tensor_labs.to(device)

            output = model(demo, med, vitals, labs)
            output = nn.Sigmoid()(output)
            output = (output > threshold).int()

            # stacking labels and predictions
            stacked_labels = torch.cat([stacked_labels, labels], dim=0, )
            stacked_preds = torch.cat([stacked_preds, output], dim=0, )

            get_list_correct_preds(output, labels, correct_preds)
            step += 1

    # calculate accuracy
    acc = np.round(np.sum(correct_preds) / len(correct_preds), 2)
    # transfer to device
    stacked_labels = stacked_labels.cpu().detach().numpy()
    stacked_preds = stacked_preds.cpu().detach().numpy()
    # get classification metrics for all samples in the test set
    classification_report_res = classification_report(stacked_labels, stacked_preds, zero_division=0, output_dict=True)
    print(classification_report(stacked_labels, stacked_preds, zero_division=0, output_dict=False))
    if log_res:
        for k_day, v_day in classification_report_res.items():
            if k_day is not 'epoch':
                for k, v in v_day.items():
                    wandb.log({"test_" + k + "_day_" + k_day : v})

    return classification_report_res, acc




# CLASS WEIGHTS
def calculate_class_weights(DF_PATH, pid_train_df):
    with open(DF_PATH + 'train_df.pkl', 'rb') as f:
        train_df = pickle.load(f)

    # get admissions of patients who had aki within first 10 days
    S = train_df[train_df.aki_status==1.0].sort_values('day_id').drop_duplicates(subset='hadm_id', keep='first')

    # adm_aki_10_days = S[S.day_id <= 10].hadm_id.to_list()
    # adm_aki_7_days = S[S.day_id <= 7].hadm_id.to_list()
    adm_aki_5_days = S[S.day_id <= 5].hadm_id.to_list()
    # adm_aki_3_days = S[S.day_id <= 3].hadm_id.to_list()

    # print('\n\nCLASS WEIGHTS')
    # do some undersampling
    pos_train_df = pid_train_df[pid_train_df.hadm_id.isin(adm_aki_5_days)]
    # print('pos_train_df.shape: ', pos_train_df.shape)

    neg_train_df = pid_train_df[~pid_train_df.hadm_id.isin(adm_aki_5_days)].sample(frac=0.5)
    # print('neg_train_df.shape: ', neg_train_df.shape)

    train_df = pd.concat([pos_train_df, neg_train_df], axis=0).sample(frac=1)
    # print('train_df.shape: ', train_df.shape)
    # print('\n')

    n_days = 5 + 1
    pos_samples = len(pos_train_df)
    neg_samples = len(pos_train_df) * (n_days - 1) + len(neg_train_df) * n_days
    n_samples = pos_samples + neg_samples

    # print(f'pos_samples: {pos_samples}')
    # print(f'neg_samples: {neg_samples}')
    # print(f'n of samples: {n_samples}')
    # print(f'ratio of pos to neg: {pos_samples / neg_samples}')

    pos_class_weight = np.round(n_samples / (2 * pos_samples), 2)
    neg_class_weight = np.round(n_samples / (2 * neg_samples), 2)

    print('\n')
    # print(f'pos_class_weight: {pos_class_weight}' )
    # print(f'neg_class_weight: {neg_class_weight}' )

    return pos_class_weight


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
          epoch_patience=6,
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
        correct_preds = []
        
        model.train()
        for (tensor_demo, tensor_med, tensor_vitals, tensor_labs), tensor_labels in train_loader:
            # transferring everything to GPU
            labels = tensor_labels.to(device)
            demo = tensor_demo.to(device)
            med = tensor_med.to(device)
            vitals = tensor_vitals.to(device)
            labs = tensor_labs.to(device)

            output = model(demo, med, vitals, labs)

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
            
            correct_preds = get_list_correct_preds(output_pred, labels, correct_preds)
            global_step += 1

            wandb.log({'step_train_loss': loss.item(), 'global_step': global_step})
            
        # calculate accuracy
        epoch_train_accuracy = np.round(np.sum(correct_preds) / len(correct_preds), 5)
        if scheduler is not None:
            scheduler.step()


        model.eval()
        correct_preds = []
        stacked_labels = torch.tensor([]).to(device)
        stacked_preds = torch.tensor([]).to(device)
        with torch.no_grad():
            # validation loop
            for (tensor_demo, tensor_med, tensor_vitals, tensor_labs), tensor_labels in valid_loader:
                labels = tensor_labels.to(device)
                demo = tensor_demo.to(device)
                med = tensor_med.to(device)
                vitals = tensor_vitals.to(device)
                labs = tensor_labs.to(device)
                
                output = model(demo, med, vitals, labs)

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
                
                
                correct_preds = get_list_correct_preds(output_pred, labels, correct_preds)

        # calculate accuracy
        epoch_val_accuracy = np.round(np.sum(correct_preds) / len(correct_preds), 5)
        # transfer to device
        stacked_labels = stacked_labels.cpu().detach().numpy()
        stacked_preds = stacked_preds.cpu().detach().numpy()
        # get classification metrics for all samples in the test set
        classification_report_res = classification_report(stacked_labels, stacked_preds, zero_division=0, output_dict=True)
        classification_report_res.update({'epoch':epoch+1})

        # log the evaluation metrics 
        for key, value in classification_report_res.items():
            wandb.log({key:value, 'epoch':epoch+1})

        # valid loss
        epoch_average_train_loss = running_loss / len(train_loader)  
        epoch_average_valid_loss = valid_running_loss / len(valid_loader)

        train_loss_list.append(epoch_average_train_loss)
        valid_loss_list.append(epoch_average_valid_loss)
        train_acc_list.append(epoch_train_accuracy)
        valid_acc_list.append(epoch_val_accuracy)


        global_steps_list.append(global_step)
        wandb.log({'epoch_average_train_loss': epoch_average_train_loss,
                    'epoch_average_valid_loss': epoch_average_valid_loss,
                    'epoch_val_accuracy': epoch_val_accuracy, 
                    'epoch_train_accuracy': epoch_train_accuracy,
                    'epoch': epoch+1})

        # resetting running values
        running_loss = 0.0                
        valid_running_loss = 0.0
        
        
        # print progress
        print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}, Valid accuracy: {:.4f}'
                .format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader),
                        epoch_average_train_loss, epoch_average_valid_loss, epoch_val_accuracy))    

        # checkpoint
        if best_valid_loss > epoch_average_valid_loss:
            best_valid_loss = epoch_average_valid_loss
            save_checkpoint(file_path + '/model.pt', model, optimizer, best_valid_loss)
            save_metrics(file_path + '/metrics_loss.pt', train_loss_list, valid_loss_list, global_steps_list)
            save_metrics(file_path + '/metrics_acc.pt', train_acc_list, valid_acc_list, global_steps_list)
        else:
            stop_training +=1
        
        if stop_training == epoch_patience:
            break


    # save_metrics(file_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')


###################### TEST #################################
def load_checkpoint(load_path, model, optimizer, device):

    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    
    return state_dict['valid_loss']




######################## MAIN ###############################

def main(PRETRAINED_PATH, saving_folder_name=None, criterion='BCELoss', small_dataset=False, use_gpu=True, project_name='test', max_days=7, pred_window=1, BATCH_SIZE=128, LR=0.0001, min_frequency=1, hidden_size=128, num_epochs=50, wandb_mode='online'):
    # define the device
    if use_gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device='cpu'

    #paths
    CURR_PATH = os.getcwd()
    PKL_PATH = CURR_PATH+'/pickles/'
    DF_PATH = CURR_PATH +'/dataframes/'
    TXT_DIR_TRAIN = CURR_PATH + '/txt_files/train'
    destination_folder = CURR_PATH + '/training/'
    
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
    embedding_size = 200
    dimension = 128
    

    # loading the data
    with open(DF_PATH + 'pid_train_df_finetuning.pkl', 'rb') as f:
        pid_train_df = pickle.load(f)

    with open(DF_PATH + 'pid_val_df_finetuning.pkl', 'rb') as f:
        pid_val_df = pickle.load(f)

    with open(DF_PATH + 'pid_test_df_finetuning.pkl', 'rb') as f:
        pid_test_df = pickle.load(f)


    if small_dataset:
        # DATALOADERS
        pid_train_df_small = pid_train_df.sample(frac=0.01)
        pid_val_df_small = pid_val_df.sample(frac=0.005)

        train_dataset = MyDataset(pid_train_df_small, tokenizer=tokenizer, max_length=max_length, max_days=max_days, pred_window=pred_window)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        val_dataset = MyDataset(pid_val_df_small, tokenizer=tokenizer, max_length=max_length, max_days=max_days, pred_window=pred_window)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

        test_dataset = MyDataset(pid_test_df, tokenizer=tokenizer, max_length=max_length, max_days=max_days, pred_window=pred_window)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    else:
        train_dataset = MyDataset(pid_train_df, tokenizer=tokenizer, max_length=max_length, max_days=max_days, pred_window=pred_window)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

        val_dataset = MyDataset(pid_val_df, tokenizer=tokenizer, max_length=max_length, max_days=max_days, pred_window=pred_window)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

        test_dataset = MyDataset(pid_test_df, tokenizer=tokenizer, max_length=max_length, max_days=max_days, pred_window=pred_window)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


    #print shapes
    (tensor_demo, tensor_med, tensor_vitals, tensor_labs), tensor_labels = next(iter(train_loader))
    print('\n\n DATA SHAPES: ')
    print('train data shape: ', pid_train_df.shape)
    print('val data shape: ', pid_val_df.shape)
    print('test data shape: ', pid_test_df.shape)

    print('tensor_demo', tensor_demo.shape)
    print('tensor_med', tensor_med.shape)
    print('tensor_vitals', tensor_vitals.shape)
    print('tensor_labs', tensor_labs.shape)
    print('tensor_labels', tensor_labels.shape)

    # initialize the pretrained model
    pretrained_model = EHR_model(embedding_size=embedding_size, vocab_size=vocab_size, max_length=max_length, pred_window=pred_window, max_day=max_days).to(device)
    pretrained_model.load_state_dict(torch.load(PRETRAINED_PATH, map_location=device)['model_state_dict'])
    pretrained_model.projection = Identity()
    pretrained_model.drop = Identity()
    print(f"Pretrained model loaded from <=== {PRETRAINED_PATH}")

    # initialize the fine-tuning model
    model = LSTM_model(max_length=max_length, H=128, max_day=max_days, pred_window=pred_window, vocab_size=vocab_size, pretrained_model=pretrained_model).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


    train_params = {
                        'model':model,
                        'device':device,
                        'optimizer':optimizer,
                        'criterion':'BCELoss',
                        'train_loader':train_loader,
                        'valid_loader':val_loader,
                        'num_epochs':num_epochs, 
                        'file_path':destination_folder,
                        'best_valid_loss':float("Inf"),
                        'dimension':128,
                        'epoch_patience':5,
                        'threshold':0.5,
                        'scheduler':exp_lr_scheduler
                    }

    if criterion=='BCEWithLogitsLoss': 
        #calculate weights
        pos_class_weight = calculate_class_weights(DF_PATH=DF_PATH, pid_train_df=pid_train_df)
        pos_weight = pos_class_weight / 3
        pos_weight = torch.ones([tensor_labels.shape[-1]]) * pos_weight
        pos_weight = pos_weight.to(device)
        train_params['pos_weight'] = pos_weight
        train_params['criterion'] = 'BCEWithLogitsLoss'
        str_weights = str(np.round(pos_weight[0].item(), 2))
    elif criterion=='BCELoss':
        # path for the model
        str_weights = ''
    else:
        print('Error: use BCELoss or BCEWithLogitsLoss. ')

    # path for the model
    if saving_folder_name is None:
        saving_folder_name = 'FT_VEC_' + str(len(train_loader)*BATCH_SIZE // 1000) + 'k_' + str_weights + '_lr'+ str(LR) + '_Adam_' + 'mf' +str(min_frequency) + '_h'+str(hidden_size) + '_days' + str(max_days) + '_pw'+str(pred_window)
    file_path = destination_folder + saving_folder_name
    train_params['file_path'] = file_path

    print(f'\n\nMODEL PATH: {file_path}')
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    run_name = str(BATCH_SIZE)+'bs' + saving_folder_name

    # wandb setup
    os.environ['WANDB_API_KEY'] = '8e859a0fc58f296096842a367ca532717d3b4059'    
    wandb.init(project=project_name, name=run_name, mode=wandb_mode)
    args = {'optimizer':optimizer, 'criterion':'BCELoss', 'max_days':max_days, 'LR':LR, 'min_frequency':min_frequency, 'hidden_size':hidden_size, 'pred_window':pred_window, 'experiment':'FT'}
    config = wandb.config
    config.update(args)

    # training
    train(**train_params)

    # testing
    print('\nTesting the model...')
    load_checkpoint(file_path + '/model.pt', model, optimizer, device=device)
    evaluate(model, test_loader, device, threshold=0.5, log_res=True)

    wandb.finish()


# test
# main(saving_folder_name='/test', criterion='BCELoss', small_dataset=True, use_gpu=True, project_name='lstm-project', max_days=7, pred_window=1, BATCH_SIZE=40, LR=0.00001, min_frequency=1, hidden_size=128,  num_epochs=2, wandb_mode='disabled')

# 27688
# main(saving_folder_name=None, criterion='BCELoss', small_dataset=False, use_gpu=True, project_name='lstm-project', max_days=7, pred_window=1, BATCH_SIZE=40, LR=0.00001, min_frequency=1, hidden_size=128,  num_epochs=20, wandb_mode='online')

# 27689
# main(saving_folder_name=None, criterion='BCELoss', small_dataset=False, use_gpu=True, project_name='lstm-project', max_days=8, pred_window=2, BATCH_SIZE=40, LR=0.00001, min_frequency=1, hidden_size=128,  num_epochs=20, wandb_mode='online')

#27690
# main(saving_folder_name=None, criterion='BCELoss', small_dataset=False, use_gpu=True, project_name='lstm-project', max_days=9, pred_window=3, BATCH_SIZE=40, LR=0.00001, min_frequency=1, hidden_size=128,  num_epochs=20, wandb_mode='online')

# # 27691
# main(saving_folder_name=None, criterion='BCELoss', small_dataset=False, use_gpu=True, project_name='lstm-project', max_days=7, pred_window=3, BATCH_SIZE=40, LR=0.00001, min_frequency=1, hidden_size=128,  num_epochs=20, wandb_mode='online')

# # 28915 temp=0.05, pw=1
# PRETRAINED_PATH = '/home/svetlana.maslenkova/LSTM/pretraining/fc1/CL_FC1_bs128_142k_lr1e-05_Adam_temp0.05/model.pt'
# main(PRETRAINED_PATH=PRETRAINED_PATH, saving_folder_name='FT_VEC_lr1e-5_Adam_mf1_h128_days7_pw1_temp0.05', criterion='BCELoss', small_dataset=False, use_gpu=True, project_name='lstm-project', max_days=7, pred_window=1, BATCH_SIZE=256, LR=0.00001, min_frequency=1, hidden_size=128,  num_epochs=20, wandb_mode='online')

# # 28916 temp=0.01, pw=1
# PRETRAINED_PATH = '/home/svetlana.maslenkova/LSTM/pretraining/fc1/CL_FC1_bs128_142k_lr1e-05_Adam_temp0.01/model.pt'
# main(PRETRAINED_PATH=PRETRAINED_PATH, saving_folder_name='FT_VEC_lr1e-5_Adam_mf1_h128_days7_pw1_temp0.01', criterion='BCELoss', small_dataset=False, use_gpu=True, project_name='lstm-project', max_days=7, pred_window=1, BATCH_SIZE=256, LR=0.00001, min_frequency=1, hidden_size=128,  num_epochs=20, wandb_mode='online')

# # 28917 temp=0.1, pw=1
# PRETRAINED_PATH = '/home/svetlana.maslenkova/LSTM/pretraining/fc1/CL_FC1_bs128_142k_lr1e-05_Adam_temp0.1/model.pt'
# main(PRETRAINED_PATH=PRETRAINED_PATH, saving_folder_name='FT_VEC_lr1e-5_Adam_mf1_h128_days7_pw1_temp0.1', criterion='BCELoss', small_dataset=False, use_gpu=True, project_name='lstm-project', max_days=7, pred_window=1, BATCH_SIZE=512, LR=0.00001, min_frequency=1, hidden_size=128,  num_epochs=20, wandb_mode='online')


# # 28918  temp=0.05, pw=2
# 28936 with scheduler
PRETRAINED_PATH = '/home/svetlana.maslenkova/LSTM/pretraining/fc1/CL_FC1_bs128_142k_lr1e-05_Adam_temp0.05/model.pt'
main(PRETRAINED_PATH=PRETRAINED_PATH, saving_folder_name='FT_VEC_lr1e-5_Adam_mf1_h128_days8_pw2_temp0.05', criterion='BCELoss', small_dataset=False, use_gpu=True, project_name='lstm-project', max_days=8, pred_window=2, BATCH_SIZE=800, LR=0.00001, min_frequency=1, hidden_size=128,  num_epochs=20, wandb_mode='online')
