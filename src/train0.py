#from curses.ascii import EM
from ast import arg
from http.client import ImproperConnectionState
import logging
import os
from statistics import mode
import time
from sklearn.model_selection import StratifiedKFold
import torch
from torch import nn
from config1 import parse_args1
from data_helper1 import create_dataloaders,create_dataloadersK
#from data_helper1 import create_dataloaders
#from inference import inference
from model1 import MultiModal
#from model1 import MultiModal
#from model2 import MultiModal2
from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate
import torch.nn.functional as F
import copy
import numpy as np
import tqdm
import gc
import pandas as pd
#ema = EMA(MultiModal(parse_args()).to('cuda'), 0.999)
#ema.register()
logger = logging.getLogger(__name__)


def validate(model, val_dataloader):
    model.eval()
    predictions = []
    labels = []
    losses = []
    #ema.apply_shadow()
    with torch.no_grad():
        for batch in val_dataloader:
            loss, _, pred_label_id, label = model(batch)
            loss = loss.mean()
            predictions.extend(pred_label_id.cpu().numpy())
            labels.extend(label.cpu().numpy())
            losses.append(loss.cpu().numpy())
    loss = sum(losses) / len(losses)
    results = evaluate(predictions, labels)
    #ema.restore()    

    model.train()
    return loss, results


def train_and_validate(args, trn_df, dev_df, K):
    # 1. load data
    #train_dataloader, val_dataloader = create_dataloaders(args)
    train_dataloader, val_dataloader = create_dataloadersK(args,trn_df,dev_df)

    # 2. build model and optimizers
    model = MultiModal(args)
    #swa_raw_model = copy.deepcopy(model)


    ema = EMA(model, 0.999)
    ema.register()
    optimizer, scheduler = build_optimizer(args, model)
    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))

    fgm = FGM(model)

    # 3. training
    step = 0
    best_score = args.best_score
    start_time = time.time()
    num_total_steps = len(train_dataloader) * args.max_epochs
    for epoch in range(args.max_epochs):
        for batch in train_dataloader:
            #正常训练
            model.train()
            loss, accuracy, _, _ = model(batch)
            #print('ttt',loss)
            loss = loss.mean()
            accuracy = accuracy.mean()
            loss.backward()

            
            #对抗训练
            fgm.attack()
            optimizer.zero_grad()
            loss_adv = model(batch)[0]
            #print('3',loss_adv)
            loss_adv.backward()
            fgm.restore()
            

            #梯度下降，更新参数
            optimizer.step()
            ema.update()
            optimizer.zero_grad()
            scheduler.step()

            step += 1
            if step % args.print_steps == 0:
                time_per_step = (time.time() - start_time) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                logging.info(f"Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f}, accuracy {accuracy:.3f}")
            
            #保存swa需要的后几个模型
            '''
            save_steps = num_total_steps // args.max_epochs
            if step % save_steps == 0:  
                save_model(args, model, step)
            '''

        # 4. validation
        ema.apply_shadow()
        loss, results = validate(model, val_dataloader)
        results = {k: round(v, 4) for k, v in results.items()}
        
        logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f}, {results}")
        
        #print('##',model.module,'##')

        # 5. save checkpoint
        mean_f1 = results['mean_f1']
        if mean_f1 > best_score:
            best_score = mean_f1
            torch.save({'epoch': epoch, 'model_state_dict': model.module.state_dict(), 'mean_f1': mean_f1},
                       f'{args.savedmodel_path}/{K}model_epoch_{epoch}.bin')
        ema.restore()

    #swa(swa_raw_model, 'data/out/', swa_start=3)

dev_df = pd.read_json('data/annotations/labeled.json')
tst_df = pd.read_json('data/annotations/test_a.json')
#dev_df['text'].fillna(' ',inplace=True)

#Kfold
def folds_train(total_df, tst_df, args):
    
    kf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    oof, tst_probs_lst = np.zeros((total_df.shape[0], args.num_workers)), []
    t_df = str(total_df['title']) + str(total_df['asr']) + str(total_df['ocr'])


    for k, (trn_idx, val_idx) in tqdm(
            enumerate(kf.split(total_df, t_df)),
            total=args.num_folds,
            desc='f-fold-training'
        ):
        trn_df, val_df = total_df.iloc[trn_idx, :], total_df.iloc[val_idx, :]
        trn_df.reset_index(drop=True, inplace=True)
        val_df.reste_index(drop=True, inplace=True)

        logging.info('-' * 32 + f'【Fold_{k + 1} Training】' + '-' * 32)
        best_model = train_and_validate(args,trn_df, dev_df)

        #val_probs = inference(val_df, best_model)
        #oof[val_idx] = val_probs

        #tst_probs = inference(tst_df, best_model)
        #tst_probs_lst.append(tst_probs)

        del trn_df, val_df, best_model
        gc.collect()
    
    return oof, tst_probs_lst

def compute_kl_loss(self, p, q, pad_mask=None):
    
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    
    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss

'''
def swa(model, model_dir, swa_start=1):
    """
    swa 滑动平均模型，一般在训练平稳阶段再使用 SWA
    """
    model_path_list = get_model_path_list(model_dir)

    assert 1 <= swa_start < len(model_path_list) - 1, \
        f'Using swa, swa start should smaller than {len(model_path_list) - 1} and bigger than 0'

    swa_model = copy.deepcopy(model)
    swa_n = 0.

    with torch.no_grad():
        for _ckpt in model_path_list[swa_start:]:
            logger.info(f'Load model from {_ckpt}')

            state_dic=torch.load(_ckpt, map_location=torch.device('cpu'))
            new_state = {}
            
            
            for k,v in state_dic.items(): #去除关键字”model"
                new_state=v
            
            for k,v in new_state.items():
                print(k)
            
            #for k,v in state_dic.items():
            #    print(v)            


            model.load_state_dict(new_state)
            
            
            tmp_para_dict = dict(model.named_parameters())

            alpha = 1. / (swa_n + 1.)

            for name, para in swa_model.named_parameters():
                para.copy_(tmp_para_dict[name].data.clone() * alpha + para.data.clone() * (1. - alpha))

            swa_n += 1

    # use 100000 to represent swa to avoid clash
    swa_model_dir = os.path.join(model_dir, f'checkpoint-100000')
    if not os.path.exists(swa_model_dir):
        os.mkdir(swa_model_dir)

    logger.info(f'Save swa model in: {swa_model_dir}')

    swa_model_path = os.path.join(swa_model_dir, 'model.bin')

    torch.save(swa_model.state_dict(), swa_model_path)

    return swa_model
'''

class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
 
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name].to('cuda')
                new_average = new_average.to('cuda')
                self.shadow[name] = new_average.clone()
 
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
 
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class FGM:
    def __init__(self, model: nn.Module, eps=1.):
        self.model = (
            model.module if hasattr(model, "module") else model
        )
        self.eps = eps
        self.backup = {}

    # only attack word embedding
    def attack(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm and not torch.isnan(norm):
                    r_at = self.eps * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        for name, para in self.model.named_parameters():
            if para.requires_grad and emb_name in name:
                assert name in self.backup
                para.data = self.backup[name]

        self.backup = {}

#ema = EMA(MultiModal(parse_args()).to('cuda'), 0.999)
#ema.register()


class PGD:
    def __init__(self, args, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}
        self.epsilon = args.epsilon
        self.emb_name = args.emb_name
        self.alpha = args.alpha

    def attack(self, is_first_attack=False):
        for name, param in self.model.bert.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, self.epsilon)

    def restore(self):
        for name, param in self.model.bert.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.bert.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.bert.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]


def main():
    args = parse_args1()
    setup_logging()
    setup_device(args)
    setup_seed(args,0)

    os.makedirs(args.savedmodel_path, exist_ok=True)
    logging.info("Training/evaluation parameters: %s", args)

    #train_and_validate(args)
    #oof, tst_probs_list  = folds_train(dev_df, tst_df, args)
    for i in range(0,10):
        trainpath = 'train_'+str(i)+'.json'
        valpath ='val_'+str(i)+'.json'
        logging.info('-' * 32 + f'【Fold_{i + 1} Training】' + '-' * 32)
        train_and_validate(args, trainpath, valpath, i)

    '''
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=args.seed)
    trn_df, val_df = None, None
    for k, (trn_idx, val_idx) in enumerate(dev_df, dev_df['category_id']):
        trn_df, val_df = dev_df.iloc[trn_idx, :], dev_df.iloc[val_idx, :]
        break
    '''

    
if __name__ == '__main__':
    main()
