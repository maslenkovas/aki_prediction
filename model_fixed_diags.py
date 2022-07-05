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
from tokenizers.pre_tokenizers import Digits, Punctuation, Whitespace
from tokenizers import pre_tokenizers
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
                labels.append(aki_status[i])

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
        tensor_labels = torch.tensor(label, dtype=torch.float64)
    

        return tensor_day, tensor_diags, tensor_labels, idx


class EHR_MODEL(nn.Module):
    def __init__(self, max_length, vocab_size, device, pred_window=2, observing_window=3,  H=128, embedding_size=200):
        super(EHR_MODEL, self).__init__()

        self.observing_window = observing_window
        self.pred_window = pred_window
        self.H = H
        self.max_length = max_length
        self.max_length_diags = 30
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.device = device

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

        self.drop = nn.Dropout(p=0.5)

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


def evaluate(model, test_loader, device, threshold=0.5, log_res=True):
    model = model.to(device)
    stacked_labels = torch.tensor([]).to(device)
    stacked_preds = torch.tensor([]).to(device)
    
    model.eval()
    step = 1
    with torch.no_grad():
        for tensor_day, tensor_diags, tensor_labels, idx in test_loader:
            # print(f'Step {step}/{len(test_loader)}' )
            labels = tensor_labels.to(device)
            day_info = tensor_day.to(device)
            tensor_diags = tensor_diags.to(device)

            output = model(day_info, tensor_diags)
            output = nn.Sigmoid()(output)
            output = (output > threshold).int()

            # stacking labels and predictions
            stacked_labels = torch.cat([stacked_labels, labels], dim=0, )
            stacked_preds = torch.cat([stacked_preds, output], dim=0, )

            step += 1

    # calculate accuracy
    acc = torch.round(torch.sum(stacked_labels==stacked_preds) / len(stacked_labels), decimals=2)
    # transfer to device
    stacked_labels = stacked_labels.cpu().detach().numpy()
    stacked_preds = stacked_preds.cpu().detach().numpy()
    # get classification metrics for all samples in the test set
    classification_report_res = classification_report(stacked_labels, stacked_preds, zero_division=0, output_dict=True)
    print(classification_report(stacked_labels, stacked_preds, zero_division=0, output_dict=False))
    if log_res:
        for k_day, v_day in classification_report_res.items():
            if k_day is not 'accuracy':
                for k, v in v_day.items():
                    if k is not 'support':
                        wandb.log({"test_" + k + k_day : v})
                        # print("test_" + k +'_'+ k_day, v)
            else:
                # print('accuracy', v_day)
                wandb.log({"test_" + k_day, v_day})

    return classification_report_res, acc



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
        epoch_train_accuracy = torch.round(torch.sum(stacked_labels==stacked_preds) / len(stacked_labels), decimals=2)
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
        # calculate accuracy
        epoch_val_accuracy = np.round(np.sum(stacked_labels==stacked_preds) / len(stacked_labels), 2)
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
            stop_training = 0
        else:
            stop_training +=1
        
        if stop_training == epoch_patience:
            break


    # save_metrics(file_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')



def main(saving_folder_name=None, criterion='BCELoss', small_dataset=False,\
     use_gpu=True, project_name='test', pred_window=2, observing_window=3, BATCH_SIZE=128, LR=0.0001,\
         min_frequency=1, hidden_size=128, num_epochs=50, wandb_mode='online', PRETRAINED_PATH=None, run_id=None):
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
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Digits(individual_digits=False), Punctuation( behavior = 'removed')])
        trainer = BpeTrainer(special_tokens=["[PAD]", "[UNK]"], min_frequency=3)
        files = glob.glob(TXT_DIR_TRAIN+'/*')
        tokenizer.train(files, trainer)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # variables for classes
    max_length = 400
    vocab_size = tokenizer.get_vocab_size()
    embedding_size = 200
    dimension = 128
    

    # loading the data
    with open(DF_PATH + 'pid_train_df_finetuning_6days_aki.pkl', 'rb') as f:
        pid_train_df = pickle.load(f)

    with open(DF_PATH + 'pid_val_df_finetuning_6days_aki.pkl', 'rb') as f:
        pid_val_df = pickle.load(f)

    with open(DF_PATH + 'pid_test_df_finetuning_6days_aki.pkl', 'rb') as f:
        pid_test_df = pickle.load(f)

    print('filtering admissions..')
    # filter the admissions

    pid_train_df = pid_train_df[pid_train_df.hadm_id.isin(train_admissions)]
    pid_val_df = pid_val_df[pid_val_df.hadm_id.isin(val_admissions)]
    pid_test_df = pid_test_df[pid_test_df.hadm_id.isin(test_admissions)]

    if small_dataset: frac=0.1
    else: frac=1

    train_dataset = MyDataset(pid_train_df.sample(frac=frac), tokenizer=tokenizer, max_length_day=400)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = MyDataset(pid_val_df.sample(frac=frac), tokenizer=tokenizer, max_length_day=400)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = MyDataset(pid_test_df.sample(frac=frac), tokenizer=tokenizer, max_length_day=400)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


    #print shapes
    # tensor_day, tensor_labels, = next(iter(train_loader))
    tensor_day, tensor_diags, tensor_labels, idx = next(iter(train_loader))
    print('\n\n DATA SHAPES: ')
    print('train data shape: ', pid_train_df.shape)
    print('val data shape: ', pid_val_df.shape)
    print('test data shape: ', pid_test_df.shape)

    print('tensor_day', tensor_day.shape)
    print('tensor_labels', tensor_labels.shape)

    # file_path = destination_folder + '/88087_no_weights-lr0.00005-adam'
    
    model = EHR_MODEL(vocab_size=vocab_size, max_length=max_length, device=device, pred_window=2, observing_window=3).to(device)

    if PRETRAINED_PATH is not None:
        model.load_state_dict(torch.load(PRETRAINED_PATH, map_location=device)['model_state_dict'])
        model.embedding.weight.data = model.embedding.weight.data
        print(f"Pretrained model loaded from <=== {PRETRAINED_PATH}")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

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
                    'epoch_patience':5,
                    'threshold':0.5,
                    'scheduler':exp_lr_scheduler
                }

    # path for the model
    if saving_folder_name is None:
        saving_folder_name = 'FX_NO_PRETRAINING_' + str(len(train_loader)*BATCH_SIZE // 1000) + 'k_'  \
            + 'lr' + str(LR) + '_h'+ str(hidden_size) + '_pw' + str(pred_window) + '_ow' + str(observing_window)
    
    file_path = destination_folder + saving_folder_name
    train_params['file_path'] = file_path

    print(f'\n\nMODEL PATH: {file_path}')
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    run_name = saving_folder_name

    # wandb setup
    os.environ['WANDB_API_KEY'] = '8e859a0fc58f296096842a367ca532717d3b4059'
    
    if run_id is None:    
        run_id = wandb.util.generate_id()  
        resume = 'allow' 
    else:
        resume = 'must'
        
    args = {'optimizer':optimizer, 'criterion':'BCELoss', 'LR':LR, 'min_frequency':min_frequency, 'hidden_size':hidden_size, 'pred_window':pred_window, 'experiment':'no_pretraining'}
    wandb.init(project=project_name, name=run_name, mode=wandb_mode, config=args, id=run_id, resume=resume)
    print('Run id is: ', run_id)

    # training
    print('Training started..')
    train(**train_params)

    # testing
    print('\nTesting the model...')
    load_checkpoint(file_path + '/model.pt', model, optimizer, device=device)
    evaluate(model, test_loader, device, threshold=0.5, log_res=True)

    wandb.finish()


#paths
print('Filtering admissions...')
CURR_PATH = os.getcwd()
PKL_PATH = CURR_PATH+'/pickles/'
DF_PATH = CURR_PATH +'/dataframes/'

# loading the data
with open(DF_PATH + 'pid_train_df_finetuning_6days_aki.pkl', 'rb') as f:
    pid_train_df = pickle.load(f)

with open(DF_PATH + 'pid_val_df_finetuning_6days_aki.pkl', 'rb') as f:
    pid_val_df = pickle.load(f)

with open(DF_PATH + 'pid_test_df_finetuning_6days_aki.pkl', 'rb') as f:
    pid_test_df = pickle.load(f)

observing_window = 3 

train_admissions = []
for adm in pid_train_df.hadm_id.unique():   
    if ({1,2,3,4}.issubset(set(pid_train_df[pid_train_df.hadm_id==adm].days.values[0])) or \
        {-1,0,1,2}.issubset(set(pid_train_df[pid_train_df.hadm_id==adm].days.values[0]))or \
            {0,1,2,3}.issubset(set(pid_train_df[pid_train_df.hadm_id==adm].days.values[0]))) and \
        (len(pid_train_df[pid_train_df.hadm_id==adm].days.values[0])>3) and\
            sum(pid_train_df[pid_train_df.hadm_id==adm].aki_status_in_visit.values[0][:observing_window])==0:
        train_admissions.append(adm)

val_admissions = []
for adm in pid_val_df.hadm_id.unique():   
    if ({1,2,3,4}.issubset(set(pid_val_df[pid_val_df.hadm_id==adm].days.values[0])) or \
        {-1,0,1,2}.issubset(set(pid_val_df[pid_val_df.hadm_id==adm].days.values[0]))or \
            {0,1,2,3}.issubset(set(pid_val_df[pid_val_df.hadm_id==adm].days.values[0]))) and \
        (len(pid_val_df[pid_val_df.hadm_id==adm].days.values[0])>3) and\
            sum(pid_val_df[pid_val_df.hadm_id==adm].aki_status_in_visit.values[0][:observing_window])==0:
        val_admissions.append(adm)

test_admissions = []
for adm in pid_test_df.hadm_id.unique():   
    if ({1,2,3,4}.issubset(set(pid_test_df[pid_test_df.hadm_id==adm].days.values[0])) or \
        {-1,0,1,2}.issubset(set(pid_test_df[pid_test_df.hadm_id==adm].days.values[0]))or \
            {0,1,2,3}.issubset(set(pid_test_df[pid_test_df.hadm_id==adm].days.values[0]))) and \
        (len(pid_test_df[pid_test_df.hadm_id==adm].days.values[0])>3) and\
            sum(pid_test_df[pid_test_df.hadm_id==adm].aki_status_in_visit.values[0][:observing_window])==0:
        test_admissions.append(adm)

print('train_admissions', len(train_admissions))
print('val_admissions', len(val_admissions))
print('test_admissions', len(test_admissions))




########################################### RUNs ############################################

main(saving_folder_name='test_model', criterion='BCELoss', small_dataset=False,\
     use_gpu=True, project_name='Fixed_obs_window_model', pred_window=2, BATCH_SIZE=128, LR=1e-05,\
         min_frequency=3, hidden_size=128, num_epochs=1, wandb_mode='disabled', PRETRAINED_PATH=None, run_id=None)