import torch
from torch.utils.data import SequentialSampler, DataLoader

#model2
from config import parse_args
from data_helper import MultiModalDataset
from category_id_map import lv2id_to_category_id
from model import MultiModal
import numpy as np

#model1
from config1 import parse_args1
from model1 import MultiModal as MultiModal1
from data_helper1 import MultiModalDataset as MultiModalDataset1
def inference():
    args = parse_args()
    # 1. load data
    dataset = MultiModalDataset(args, args.test_annotation, args.test_zip_feats, test_mode=True)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=args.test_batch_size,
                            sampler=sampler,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=args.num_workers,
                            prefetch_factor=args.prefetch)
    num =  10
    # 2. load model
    pred = []
    for i in range(num):
        print('第%d个'%i)
        model = MultiModal(args)
        if i == 0:
            checkpoint = torch.load(args.ckpt_file_0 , map_location='cpu')
        elif i == 1:
            checkpoint = torch.load(args.ckpt_file_1, map_location='cpu')
        elif i == 2:
            checkpoint = torch.load('save/v1/model_epoch_4_fold1.bin', map_location='cpu')
        elif i == 3:
            checkpoint = torch.load('save/v1/model_epoch_4_fold2.bin', map_location='cpu')
        elif i == 4:
            checkpoint = torch.load('save/v1/model_epoch_4_fold3.bin', map_location='cpu')
        elif i == 5:
            checkpoint = torch.load('save/v1/model_epoch_4_fold4.bin', map_location='cpu')
        elif i == 6:
            checkpoint = torch.load('save/v1/model_epoch_4_fold5.bin', map_location='cpu')
        elif i == 7:
            checkpoint = torch.load('save/v1/model_epoch_4_fold6.bin', map_location='cpu')
        elif i == 8:
            checkpoint = torch.load('save/v1/model_epoch_4_fold7.bin', map_location='cpu')
        elif i == 9:
            checkpoint = torch.load('save/v1/model_epoch_4_fold8.bin', map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])

        if torch.cuda.is_available():
            model = torch.nn.parallel.DataParallel(model.cuda())
        model.eval()

        # def multiply_2D_list(l, by=1.05):
        #     return [[i * by for i in sub_list] for sub_list in l]



        # 3. inference
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                pred_label_id = model(batch, inference1=True)
                predictions.extend(pred_label_id.cpu().numpy())
        pred = np.sum([ pred,predictions],axis=0)
        # print(pred )

    args1 = parse_args1()
    # 1. load data
    dataset1 = MultiModalDataset1(args1, args1.test_annotation, args1.test_zip_feats, test_mode=True)
    sampler1 = SequentialSampler(dataset1)
    dataloader1 = DataLoader(dataset1,
                            batch_size=args1.test_batch_size,
                            sampler=sampler1,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=args1.num_workers,
                            prefetch_factor=args1.prefetch)
    num2 = 17
    # 2. load model
    for i in range(num,num2):
        print('第%d个' % i)
        model1 = MultiModal1(args1)
        #model_epoch_{epoch}.bin
        if i == 10:
            checkpoint = torch.load('save/m1/0model_epoch_4.bin', map_location='cpu')
        elif i == 11:
            checkpoint = torch.load('save/m1/1model_epoch_4.bin', map_location='cpu')
        elif i == 12:
            checkpoint = torch.load('save/m1/3model_epoch_3.bin', map_location='cpu')
        elif i == 13:
            checkpoint = torch.load('save/m1/6model_epoch_3.bin', map_location='cpu')
        elif i == 14:
            checkpoint = torch.load('save/m1/7model_epoch_3.bin', map_location='cpu')
        elif i == 15:
            checkpoint = torch.load('save/m1/8model_epoch_4.bin', map_location='cpu')
        elif i == 16:
            checkpoint = torch.load('save/m1/9model_epoch_4.bin', map_location='cpu')
        model1.load_state_dict(checkpoint['model_state_dict'])

        if torch.cuda.is_available():
            model1 = torch.nn.parallel.DataParallel(model1.cuda())
        model1.eval()

        # 3. inference
        predictions = []
        with torch.no_grad():
            for batch in dataloader1:
                pred_label_id = model1(batch, inference1=True)
                predictions.extend(pred_label_id.cpu().numpy())
        pred = np.sum([pred, predictions], axis=0)


    pred = pred/num2
    predictionsl = np.argmax(pred,axis=1)

    # predictions = torch.argmax(predictions,dim=1)


    # 4. dump results
    with open(args.test_output_csv, 'w') as f:
        for pred_label_id, ann in zip(predictionsl, dataset.anns):
            video_id = ann['id']
            category_id = lv2id_to_category_id(pred_label_id)
            f.write(f'{video_id},{category_id}\n')



if __name__ == '__main__':
    inference()
