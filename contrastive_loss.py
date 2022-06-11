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
import os
import pickle5 as pickle
import wandb



# CLASSES
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
        b, m, n = representations.size()
        representations = representations.reshape(b, m*n)
        
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask.to(self.device) * torch.exp(similarity_matrix / self.temperature)
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss



class MyDataset(Dataset):

    def __init__(self, df, tokenizer, max_length):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len_demo = max_length['demographics']
        self.max_len_labs = max_length['lab_tests']
        self.max_len_vitals = max_length['vitals']
        self.max_len_meds = max_length['medications']
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):

        return self.make_matrices(idx)
    
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

    def make_matrices(self, idx):
        info_demo = self.df.demographics.values[idx]
        info_med = self.df.medications.values[idx]
        info_vitals = self.df.vitals.values[idx]
        info_labs = self.df.lab_tests.values[idx]
        
        if str(info_med) == 'nan':
            info_med = self.tokenize('[PAD]', self.max_len_meds)
        else:
            info_med = self.tokenize(info_med,  self.max_len_meds)
        if str(info_demo) == 'nan':
            info_demo = self.tokenize('[PAD]', self.max_len_demo)
        else:
            info_demo = self.tokenize(info_demo,  self.max_len_demo)
        if str(info_vitals) == 'nan':
            info_vitals = self.tokenize('[PAD]', self.max_len_vitals)
        else:
            info_vitals = self.tokenize(info_vitals,  self.max_len_vitals)
        if str(info_labs) == 'nan':
            info_labs = self.tokenize('[PAD]', self.max_len_labs)
        else:
            info_labs = self.tokenize(info_labs,  self.max_len_labs)
        
        #make tensors
        tensor_demo = torch.tensor(info_demo, dtype=torch.float32)
        tensor_med = torch.tensor(info_med, dtype=torch.float32)
        tensor_vitals = torch.tensor(info_vitals, dtype=torch.float32)
        tensor_labs = torch.tensor(info_labs, dtype=torch.float32)

        # t1 = {
        #     'tensor_demo': self.drop(tensor_demo).type(torch.int32),
        #     'tensor_med': self.drop(tensor_med).type(torch.int32),
        #     'tensor_vitals' : self.drop(tensor_vitals).type(torch.int32),
        #     'tensor_labs' : self.drop(tensor_labs).type(torch.int32)
        # }
        # t2 = {
        #     'tensor_demo' : self.drop(tensor_demo).type(torch.int32),
        #     'tensor_med' : self.drop(tensor_med).type(torch.int32),
        #     'tensor_vitals' : self.drop(tensor_vitals).type(torch.int32),
        #     'tensor_labs' : self.drop(tensor_labs).type(torch.int32)
        # }

        t1 = {
            'tensor_demo': tensor_demo.type(torch.int32),
            'tensor_med': tensor_med.type(torch.int32),
            'tensor_vitals' : tensor_vitals.type(torch.int32),
            'tensor_labs' : tensor_labs.type(torch.int32)
        }
        t2 = {
            'tensor_demo' : tensor_demo.type(torch.int32),
            'tensor_med' : tensor_med.type(torch.int32),
            'tensor_vitals' : tensor_vitals.type(torch.int32),
            'tensor_labs' : tensor_labs.type(torch.int32)
        }

        return t1, t2


class EHR_Embedding(nn.Module):
    def __init__(self, embedding_size, vocab_size=15463, drop=0.1):
        super(EHR_Embedding, self).__init__()
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.projection = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_features=embedding_size, out_features=embedding_size)
        )
        self.drop = nn.Dropout(p=drop)
        
    def forward(self, tensor_demo, tensor_med, tensor_vitals, tensor_labs):   
        batch_size = tensor_med.size()[0]

        # first traansformation
        emb_demo_X = self.drop(self.embedding(tensor_demo.squeeze(1)))
        emb_med_X = self.drop(self.embedding(tensor_med[:,:].squeeze(1)))
        emb_vitals_X = self.drop(self.embedding(tensor_vitals[:,:].squeeze(1)))
        emb_labs_X =  self.drop(self.embedding(tensor_labs[:,:].squeeze(1)))

        projection_demo_X = self.projection(emb_demo_X)
        projection_med_X = self.projection(emb_med_X)
        projection_vitals_X = self.projection(emb_vitals_X)
        projection_labs_X = self.projection(emb_labs_X)

        embedding_X = (emb_demo_X, emb_med_X, emb_vitals_X, emb_labs_X)
        projection_X = (projection_demo_X, projection_med_X, projection_vitals_X, projection_labs_X)

        # second transformation
        emb_demo_Y = self.drop(self.embedding(tensor_demo.squeeze(1)))
        emb_med_Y = self.drop(self.embedding(tensor_med[:,:].squeeze(1)))
        emb_vitals_Y = self.drop(self.embedding(tensor_vitals[:,:].squeeze(1)))
        emb_labs_Y =  self.drop(self.embedding(tensor_labs[:,:].squeeze(1)))

        projection_demo_Y = self.projection(emb_demo_Y)
        projection_med_Y = self.projection(emb_med_Y)
        projection_vitals_Y = self.projection(emb_vitals_Y)
        projection_labs_Y = self.projection(emb_labs_Y)

        embedding_Y = (emb_demo_Y, emb_med_Y, emb_vitals_Y, emb_labs_Y)
        projection_Y = (projection_demo_Y, projection_med_Y, projection_vitals_Y, projection_labs_Y)

        return embedding_X, projection_X, embedding_Y, projection_Y
 
 


# FUNCTIONS
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
          epoch_patience=8,
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

            for X, Y in train_loader:
                if train_step % 10000==0:
                    print(f'Step {train_step}/{total_train_steps}')
                d = 0
                

                embX, projectionX, embY, projectionY = model(X['tensor_demo'].to(device),
                                                            X['tensor_med'].to(device),
                                                            X['tensor_vitals'].to(device),
                                                            X['tensor_labs'].to(device))
                                        
                if train_step >= total_train_steps:
                    new_batch_size = projectionX[0].size()[0]
                    criterion = ContrastiveLoss(batch_size=new_batch_size, device=device)

                loss0 = criterion(projectionX[0].type(torch.float32), projectionY[0].type(torch.float32))
                loss1 = criterion(projectionX[1].type(torch.float32), projectionY[1].type(torch.float32))
                loss2 = criterion(projectionX[2].type(torch.float32), projectionY[2].type(torch.float32))
                loss3 = criterion(projectionX[3].type(torch.float32), projectionY[3].type(torch.float32))
                loss = loss0+loss1+loss2+loss3

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
                        embX, projectionX, embY, projectionY = model(X['tensor_demo'].to(device),
                                                                        X['tensor_med'].to(device),
                                                                        X['tensor_vitals'].to(device),
                                                                        X['tensor_labs'].to(device))

                        if val_step >= total_val_steps:
                              new_batch_size = projectionX[0].size()[0]
                              criterion = ContrastiveLoss(batch_size=new_batch_size, device=device)
                                               
                        loss0 = criterion(projectionX[0].type(torch.float32), projectionY[0].type(torch.float32))
                        loss1 = criterion(projectionX[1].type(torch.float32), projectionY[1].type(torch.float32))
                        loss2 = criterion(projectionX[2].type(torch.float32), projectionY[2].type(torch.float32))
                        loss3 = criterion(projectionX[3].type(torch.float32), projectionY[3].type(torch.float32))
                        loss = loss0+loss1+loss2+loss3

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



# MAIN FUNCTION
import os
import pickle5 as pickle

def main(project_name, num_epochs, drop=0.1, embedding_size=150, min_frequency=1, BATCH_SIZE=16, small_dataset=True, LR=0.00001, save_model=False, use_gpu=True, saving_folder_name=None, log_results=False):
    
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
    destination_folder = CURR_PATH + '/pretraining/embeddings/'


    # Training the tokenizer
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=["[PAD]", "[UNK]"], min_frequency=min_frequency)
    files = glob.glob(TXT_DIR_TRAIN+'/*')
    tokenizer.train(files, trainer)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # variables for classes
    max_length = {'demographics':5, 'lab_tests':400, 'vitals':200, 'medications':255}
    vocab_size = tokenizer.get_vocab_size()
    print(f'Vocab size: {vocab_size}')

    with open(DF_PATH + 'train_df_pretraining.pkl', 'rb') as f:
        pid_train_df = pickle.load(f)

    with open(DF_PATH + 'val_df_pretraining.pkl', 'rb') as f:
        pid_val_df = pickle.load(f)

    if small_dataset:
        pid_train_df_small = pid_train_df.sample(frac=0.0001)
        pid_val_df_small = pid_val_df.sample(frac=0.005)

        train_dataset = MyDataset(pid_train_df_small, tokenizer=tokenizer, max_length=max_length)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        val_dataset = MyDataset(pid_val_df_small, tokenizer=tokenizer, max_length=max_length)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    else:
        train_dataset = MyDataset(pid_train_df, tokenizer=tokenizer, max_length=max_length)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        val_dataset = MyDataset(pid_val_df, tokenizer=tokenizer, max_length=max_length)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print('DATA SHAPES: ')
    print('train data shape: ', len(train_loader)*BATCH_SIZE)
    print('val data shape: ', len(val_loader)*BATCH_SIZE)

    model = EHR_Embedding(vocab_size=vocab_size, embedding_size=embedding_size, drop=0.1).to(device)
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
        saving_folder_name = 'CL_EMB_' + str(num_samples) + 'k' + '_lr'+ str(LR) + '_Adam'
    file_path = destination_folder + saving_folder_name
    train_params['file_path'] = file_path

    print(f'\n\nMODEL PATH: {file_path}')
    if not os.path.exists(file_path):
        os.makedirs(file_path)
        
    run_name = saving_folder_name

    args = {'optimizer':optimizer, 'LR':LR, 'min_frequency':min_frequency, 'dropout':drop, 'vocab_size':vocab_size, 'embedding_size':embedding_size, 'pretrained':'embeddings'}

    if log_results:
        wandb.init(project=project_name, name=run_name)
        train(**train_params)
        wandb.finish()
    else:
        train(**train_params)

# 26775
# main(project_name='Contrastive-loss-pretraining', num_epochs=15, embedding_size=200, min_frequency=1, BATCH_SIZE=32, small_dataset=False, LR=0.00001, save_model=True, saving_folder_name=None, use_gpu=True, log_results=True)

# 
main(project_name='Contrastive-loss-pretraining', num_epochs=15, embedding_size=200, min_frequency=1, BATCH_SIZE=128, small_dataset=True, LR=0.00001, save_model=True, saving_folder_name=None, use_gpu=False, log_results=False)