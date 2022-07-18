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
from sklearn.metrics import classification_report, precision_recall_curve
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
        # tensor_labels = torch.tensor(label, dtype=torch.float64)
    
        return tensor_day, tensor_diags, tensor_labels, idx

########################################### MODEL #################################################

# Pretraining embedding model
class EHR_Embedding(nn.Module):
    def __init__(self, embedding_size, vocab_size, drop=0.1):
        super(EHR_Embedding, self).__init__()
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.projection = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_features=embedding_size, out_features=embedding_size)
        )
        self.drop = nn.Dropout(p=drop)
        
    def forward(self, tensor_day, tensor_diagnoses):   
        batch_size = tensor_day.size()[0]

        # first traansformation
        emb_diags_X = self.drop(self.embedding(tensor_diagnoses.squeeze(1)))
        emb_day_info_X = self.drop(self.embedding(tensor_day.squeeze(1)))

        projection_diags_X = self.projection(emb_diags_X)
        projection_day_info_X = self.projection(emb_day_info_X)

        embedding_X = (emb_diags_X, emb_day_info_X)
        projection_X = (projection_diags_X, projection_day_info_X)

        # second transformation
        emb_diags_Y = self.drop(self.embedding(tensor_diagnoses.squeeze(1)))
        emb_day_info_Y = self.drop(self.embedding(tensor_day.squeeze(1)))

        projection_diags_Y = self.projection(emb_diags_Y)
        projection_day_info_Y = self.projection(emb_day_info_Y)

        embedding_Y = (emb_diags_Y, emb_day_info_Y)
        projection_Y = (projection_diags_Y, projection_day_info_Y)

        return embedding_X, projection_X, embedding_Y, projection_Y


# The fine-tuning model
class EHR_MODEL(nn.Module):
    def __init__(self, max_length, vocab_size, device, pred_window=2, observing_window=3,  H=128, embedding_size=200, drop=0.6):
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

        self.fc_2 = nn.Linear(self.H*2, 2)

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
            output_lstm_day= self.drop(self.fc_day(\
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
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# def calculate_class_weights(data_loader):
#     i = 0
#     for tensor_day, tensor_diags, tensor_labels, idx in data_loader:
#         if i == 0:
#             labels = np.array([]).reshape(0, tensor_labels.size(-1))
#         labels = np.concatenate([labels, tensor_labels], axis=0, )
#         i += 1
#     train_stacked_labels = labels.T
#     n_pos = np.sum(train_stacked_labels, axis=1)
#     n_neg = train_stacked_labels.shape[1] - n_pos
#     weights = np.round(n_neg / n_pos, 2)
    
#     return weights

def evaluate(model, test_loader, device, use_sigmoid, log_res=True):
    model = model.to(device)
    stacked_labels = torch.tensor([]).to(device)
    # stacked_preds = torch.tensor([]).to(device)
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
            if use_sigmoid:
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
    # stacked_preds = stacked_preds.cpu().detach().numpy()

    ### get the best threshold to evaluate ###
    precision, recall, thresholds = precision_recall_curve(stacked_labels, stacked_probs)
    # convert to f score
    fscore = (2 * precision * recall) / (precision + recall)
    # locate the index of the largest f score
    ix = np.argmax(np.nan_to_num(fscore))
    print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))
    threshold = np.round(thresholds[ix], 2)
    stacked_preds = (stacked_probs > threshold).astype(int)

    # calculate accuracy
    acc = torch.round(torch.sum(stacked_labels==stacked_preds) / len(stacked_labels), decimals=2)

    # get classification metrics for all samples in the test set
    classification_report_res = classification_report(stacked_labels, stacked_preds, zero_division=0, output_dict=True)
    print(classification_report(stacked_labels, stacked_preds, zero_division=0, output_dict=False))
    if log_res:
        for k_day, v_day in classification_report_res.items():
            if k_day != 'accuracy':
                for k, v in v_day.items():
                    if k != 'support':
                        wandb.log({"test_" + k + k_day : v})
                        # print("test_" + k +'_'+ k_day, v)
            else:
                # print('accuracy', v_day)
                wandb.log({"test_" + k_day: v_day})

    # plot PR Curve
    # plot the roc curve for the model
    no_skill = len(stacked_labels[stacked_labels==1]) / len(stacked_labels)
    plt.plot([0,1], [no_skill,no_skill], linestyle='--', label='No Skill')
    plt.plot(recall, precision, marker='.', label='Model')
    plt.scatter(recall[ix], precision[ix], marker='o', color='black', label='Best')
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve')
    plt.legend()
    # show the plot
    plt.show()
    wandb.log({"chart": plt})

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
        # for key, value in classification_report_res.items():
        #     wandb.log({key:value, 'epoch':epoch+1})

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



######################## MAIN ###############################
def main(saving_folder_name=None, criterion='BCELoss', small_dataset=False,\
     use_gpu=True, project_name='test', experiment='test', pred_window=2, observing_window=3, weight_decay=0, BATCH_SIZE=128, LR=0.0001,\
         min_frequency=1, hidden_size=128, drop=0.6, num_epochs=50, wandb_mode='online', PRETRAINED_PATH=None, run_id=None):
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
    destination_folder = '/l/users/svetlana.maslenkova/models' + '/finetuning/embeddings/'
    # destination_folder = '/home/svetlanamaslenkova/Documents/AKI_deep/LSTM/training/'

    print(f'Current working directory is {CURR_PATH}')


    # Training the tokenizer
    if exists(CURR_PATH + '/tokenizer.json'):
        tokenizer = Tokenizer.from_file(CURR_PATH + '/tokenizer.json')
        print(f'Tokenizer is loaded from ==> {CURR_PATH}/tokenizer.json. Vocab size is {tokenizer.get_vocab_size()}')
    else:
        print('Training tokenizer...')
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(special_tokens=["[PAD]", "[UNK]"], min_frequency=min_frequency)
        files = glob.glob(TXT_DIR_TRAIN+'/*')
        tokenizer.train(files, trainer)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        print(f'Vocab size is {tokenizer.get_vocab_size()}')

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


    # pid_train_df = pid_train_df[pid_train_df.hadm_id.isin(train_admissions)]
    # pid_val_df = pid_val_df[pid_val_df.hadm_id.isin(val_admissions)]
    # pid_test_df = pid_test_df[pid_test_df.hadm_id.isin(test_admissions)]

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
    model = EHR_MODEL(vocab_size=vocab_size, max_length=max_length, device=device, pred_window=2, observing_window=3, drop=drop).to(device)
    if PRETRAINED_PATH is not None:
        pretrained_model = EHR_Embedding(vocab_size=vocab_size, embedding_size=embedding_size, drop=drop).to(device)
        pretrained_model.load_state_dict(torch.load(PRETRAINED_PATH, map_location=device)['model_state_dict'])
        print(f"Pretrained model loaded from <=== {PRETRAINED_PATH}")
        with torch.no_grad():
            pretrained_model.embedding.weight.copy_(model.embedding.weight)

        for name, param in model.named_parameters():
            if name == 'embedding.weight':
                param.requires_grad = False
                print(f'{name}.requires_grad is set to False')


    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
    # Decay LR by a factor of 0.1 every 7 epochs
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[15,30, 50], gamma=0.1)
    # exp_lr_scheduler = None

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
                    'threshold':0.5,
                    'scheduler':exp_lr_scheduler
                }

    weights = ''
    use_sigmoid = True
    if criterion=='BCEWithLogitsLoss':
        #calculate weights
        print('Calculating class weights..')
        pos_weight = torch.tensor(calculate_class_weights(train_loader))
        print(f'Calss weights are {pos_weight}')
        pos_weight = pos_weight.to(device)
        train_params['pos_weight'] = pos_weight
        weights = 'with_weights'
        use_sigmoid = False


    # path for the model
    if saving_folder_name is None:
        saving_folder_name = 'FT_PREemb_DIAGS_' + str(len(train_loader)*BATCH_SIZE // 1000) + 'k_'  \
            + 'lr' + str(LR) + '_h'+ str(hidden_size) + '_pw' + str(pred_window) + '_ow' + str(observing_window) + '_wd' + str(weight_decay) + '_'+ weights + '_drop' + str(drop)
    
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
        
    args = {'optimizer':optimizer, 'criterion':'BCELoss', 'LR':LR, 'min_frequency':min_frequency, 'hidden_size':hidden_size, 'pred_window':pred_window, 'experiment':experiment, 'weight_decay':weight_decay, 'drop':drop}
    wandb.init(project=project_name, name=run_name, mode=wandb_mode, config=args, id=run_id, resume=resume)
    print('Run id is: ', run_id)

    # training
    print('Training started..')
    train(**train_params)

    # testing
    print('\nTesting the model...')
    load_checkpoint(file_path + '/model.pt', model, optimizer, device=device)
    evaluate(model, test_loader, device, use_sigmoid, log_res=True)

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
PRETRAINED_PATH = '/l/users/svetlana.maslenkova/models/pretraining/embeddings/CL_EMB_FX_DIAGS_200_bs64_2065k_lr1e-05_Adam_temp0.1/model.pt'
main(saving_folder_name='test_model', criterion='BCEWithLogitsLoss', small_dataset=True,\
     use_gpu=False, project_name='test', pred_window=2, weight_decay=0, BATCH_SIZE=128  , LR=1e-05,\
         min_frequency=5, hidden_size=128, drop=0.4, num_epochs=1, wandb_mode='online', PRETRAINED_PATH=PRETRAINED_PATH, run_id=None)


# 35172:
# PRETRAINED_PATH = '/l/users/svetlana.maslenkova/models/pretraining/embeddings/CL_EMB_FX_DIAGS_200_bs64_2065k_lr1e-05_Adam_temp0.1/model.pt'
# main(saving_folder_name=None, criterion='BCEWithLogitsLoss', small_dataset=False,\
#      use_gpu=True, project_name='Fixed_obs_window_model', experiment='pretrained_embeddings', pred_window=2, weight_decay=0, BATCH_SIZE=1024  , LR=1e-05,\
#          min_frequency=5, hidden_size=128, drop=0.4, num_epochs=100, wandb_mode='online', PRETRAINED_PATH=PRETRAINED_PATH, run_id=None)

#  35174
# PRETRAINED_PATH = '/l/users/svetlana.maslenkova/models/pretraining/embeddings/CL_EMB_FX_DIAGS_200_bs64_2065k_lr1e-05_Adam_temp0.1/model.pt'
# main(saving_folder_name=None, criterion='BCELoss', small_dataset=False,\
#      use_gpu=True, project_name='Fixed_obs_window_model', experiment='pretrained_embeddings', pred_window=2, weight_decay=0, BATCH_SIZE=1024  , LR=1e-05,\
#          min_frequency=5, hidden_size=128, drop=0.4, num_epochs=100, wandb_mode='online', PRETRAINED_PATH=PRETRAINED_PATH, run_id=None)