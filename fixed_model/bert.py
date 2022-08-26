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
from transformers import AutoTokenizer, AutoModel



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




from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_length_day=400, diags='icd', pred_window=2, observing_window=2):
        self.df = dataframe
        self.tokenizer = tokenizer
        self.observing_window = observing_window
        self.pred_window = pred_window
        self.max_length_day = max_length_day
        self.diags = diags

        if self.diags == 'titles':
            self.max_length_diags = 400
        else:
            self.max_length_diags = 30
        
        self.max_len = 512

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        hadm_id = self.df.hadm_id.values[idx]
        if self.diags == 'titles':
            diagnoses_info = self.df.previous_diags_titles.values[idx][0]
        else:
            diagnoses_info = self.df.previous_diags_icd.values[idx][0]
        
        day_info = self.df.days_in_visit.values[idx]
        days = self.df.days.values[idx]
        AKI_2_status = self.df.AKI_2_in_visit.values[idx]
        AKI_3_status = self.df.AKI_3_in_visit.values[idx]
        day_info_list = []
        AKI_2_labels = []
        AKI_3_labels = []

        for day in range(0, self.observing_window + self.pred_window):

            if day not in days:
                day_info_list.append('')
                AKI_2_labels.append(0)
                AKI_3_labels.append(0)
            else:
                i = days.index(day)

                if np.isfinite(AKI_2_status[i]):                    
                    AKI_2_labels.append(AKI_2_status[i])
                else:
                    AKI_2_labels.append(0)

                if np.isfinite(AKI_3_status[i]):                    
                    AKI_3_labels.append(AKI_3_status[i])
                else:
                    AKI_3_labels.append(0)

                if (str(day_info[i]) == 'nan') or (day_info[i] == np.nan):
                    day_info_list.append('')
                else:
                    day_info_list.append(day_info[i])
        # diagnoses
        if (str(diagnoses_info) == 'nan') or (diagnoses_info == np.nan):
            diagnoses_info = '' + '$'
        else:
            diagnoses_info = diagnoses_info + '$'

        if sum(AKI_3_labels[-self.pred_window:]) > 0:
            AKI_2 = 1
            AKI_3 = 1
        elif sum(AKI_2_labels[-self.pred_window:]) > 0:
            AKI_2 = 1
            AKI_3 = 0
        else:
            AKI_2 = 0
            AKI_3 = 0
        
        self.text = ' '.join([*[diagnoses_info], *day_info_list[:self.observing_window]]).lower()

        inputs = self.tokenizer.encode_plus(
            self.text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'labels': torch.tensor([AKI_2, AKI_3], dtype=torch.float64)
        }


class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.l2 = torch.nn.Dropout(0.3)
        self.lstm = nn.LSTM(input_size=768,
                            hidden_size=128,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.l3 = torch.nn.Linear(256, 2)
    
    def forward(self, ids, mask, token_type_ids):
        output = self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids)
        output = self.l2(output[1]).unsqueeze(1)
        output, _ = self.lstm(output)
        output = self.l3(output).squeeze(1)
        return output


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
        for data in train_loader:
            # transferring everything to GPU
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            labels = data['labels'].to(device, dtype = torch.float)

            output = model(ids, mask, token_type_ids)

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
            for data in valid_loader:
                ids = data['ids'].to(device, dtype = torch.long)
                mask = data['mask'].to(device, dtype = torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
                labels = data['labels'].to(device, dtype = torch.float)
                
                output = model(ids, mask, token_type_ids)

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
            wandb.log({'val_f1_score_' + stage: f1_score_, 'val_recall_score_'+stage:recall_score_, \
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
        for data in test_loader:
            print(f'Step {step}/{len(test_loader)}' )

            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            labels = data['labels'].to(device, dtype = torch.float)

            probs = model(ids, mask, token_type_ids)
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


def main(saving_folder_name=None, additional_name='', criterion='BCELoss', small_dataset=False,\
    use_gpu=True, project_name='test', experiment='test', oversampling=False, diagnoses='icd',\
            pred_window=2, observing_window=2, BATCH_SIZE=128, LR=0.0001, min_frequency=1, hidden_size=128,\
                drop=0.6, weight_decay=0, num_epochs=1, wandb_mode='disabled', PRETRAINED_PATH=None, run_id=None):
    # define the device
    if use_gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device='cpu'
    print(f'Device: {device}')

    #paths
    CURR_PATH = os.getcwd() #+ '/LSTM/'
    PKL_PATH = CURR_PATH+'/pickles/'
    DF_PATH = CURR_PATH +'/dataframes/'
    
    destination_folder = CURR_PATH + '/training/'

    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)

    # variables for classes
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

    print('filtering admissions..')
    # filter the admissions

    if small_dataset: frac=0.1
    else: frac=1

    train_dataset = MyDataset(pid_train_df.sample(frac=frac), tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = MyDataset(pid_val_df.sample(frac=frac), tokenizer=tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = MyDataset(pid_test_df.sample(frac=frac), tokenizer=tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    if oversampling:
        i = 0
        for tensor_day, tensor_diags, tensor_labels, hadm_id in train_loader:
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

    # file_path = destination_folder + '/88087_no_weights-lr0.00005-adam'
    
    ft_model = BERTClass().to(device)
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

    # path for the model
    if saving_folder_name is None:
        saving_folder_name = additional_name + 'FT_' + experiment + '_' + str(diagnoses) + str(len(train_dataset) // 1000) + 'k_'  \
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
        
    args = {'optimizer':'Adam', 'criterion':'BCELoss', 'LR':LR, 'min_frequency':min_frequency, 'hidden_size':hidden_size, \
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
    evaluate(model, test_loader, device, threshold=None, log_res=True)

    wandb.finish()


# # test run
main(saving_folder_name='test_model', criterion='BCELoss', small_dataset=False,\
     use_gpu=True, project_name='test', experiment='biobert', oversampling=False,\
        diagnoses= 'icd', pred_window=2, weight_decay=0, BATCH_SIZE=2, LR=1e-04, min_frequency=10, hidden_size=128,\
             drop=0.6, num_epochs=1, wandb_mode='disabled', PRETRAINED_PATH=None, run_id=None)
