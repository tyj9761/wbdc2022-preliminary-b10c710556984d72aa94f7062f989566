import json
import random
import zipfile
from io import BytesIO
from functools import partial

import numpy as np
from sklearn.exceptions import DataDimensionalityWarning
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import BertTokenizer

from category_id_map import category_id_to_lv2id


def create_dataloaders(args):
    dataset = MultiModalDataset(args, args.train_annotation, args.train_zip_feats)
    size = len(dataset)
    val_size = int(size * args.val_ratio)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [size - val_size, val_size],
                                                               generator=torch.Generator().manual_seed(args.seed))

    if args.num_workers > 0:
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=args.num_workers, prefetch_factor=args.prefetch)
    else:
        # single-thread reading does not support prefetch_factor arg
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)

    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)
    train_dataloader = dataloader_class(train_dataset,
                                        batch_size=args.batch_size,
                                        sampler=train_sampler,
                                        drop_last=True)
    val_dataloader = dataloader_class(val_dataset,
                                      batch_size=args.val_batch_size,
                                      sampler=val_sampler,
                                      drop_last=False)
    return train_dataloader, val_dataloader

def create_dataloadersK(args, train_df, val_df):
    trn_dataset = MultiModalDataset(args, train_df, args.train_zip_feats)
    val_dataset = MultiModalDataset(args, val_df, args.train_zip_feats)

    if args.num_workers > 0:
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=args.num_workers, prefetch_factor=args.prefetch) 
    else:
        #single-thread reading does not support prefetch_factor
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)

    train_sampler = RandomSampler(trn_dataset, replacement=False, )       
    val_sampler = SequentialSampler(val_dataset, )

    train_dataloader = dataloader_class(trn_dataset,
                                        batch_size=args.batch_size,
                                        sampler=train_sampler,
                                        drop_last=False)
    val_dataloader = dataloader_class(val_dataset,
                                        batch_size=args.val_batch_size,
                                        sampler=val_sampler,
                                        drop_last=False)
    return train_dataloader, val_dataloader


class MultiModalDataset(Dataset):
    """ A simple class that supports multi-modal inputs.
    For the visual features, this dataset class will read the pre-extracted
    features from the .npy files. For the title information, it
    uses the BERT tokenizer to tokenize. We simply ignore the ASR & OCR text in this implementation.
    Args:
        ann_path (str): annotation file path, with the '.json' suffix.
        zip_feats (str): visual feature zip file path.
        test_mode (bool): if it's for testing.
    """

    def __init__(self,
                 args,
                 ann_path: str,
                 zip_feats: str,
                 test_mode: bool = False):
        self.max_frame = args.max_frames
        self.bert_seq_length = args.bert_seq_length
        self.test_mode = test_mode

        self.zip_feat_path = zip_feats
        self.num_workers = args.num_workers
        if self.num_workers > 0:
            # lazy initialization for zip_handler to avoid multiprocessing-reading error
            self.handles = [None for _ in range(args.num_workers)]
        else:
            self.handles = zipfile.ZipFile(self.zip_feat_path, 'r')
        # load annotations
        with open(ann_path, 'r', encoding='utf8') as f:
            self.anns = json.load(f)
        # initialize the text tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_dir, use_fast=True, cache_dir=args.bert_cache)

    def __len__(self) -> int:
        return len(self.anns)

    def get_visual_feats(self, idx: int) -> tuple:
        # read data from zipfile
        vid = self.anns[idx]['id']
        if self.num_workers > 0:
            worker_id = torch.utils.data.get_worker_info().id
            if self.handles[worker_id] is None:
                self.handles[worker_id] = zipfile.ZipFile(self.zip_feat_path, 'r')
            handle = self.handles[worker_id]
        else:
            handle = self.handles
        raw_feats = np.load(BytesIO(handle.read(name=f'{vid}.npy')), allow_pickle=True)
        raw_feats = raw_feats.astype(np.float32)  # float16 to float32
        num_frames, feat_dim = raw_feats.shape

        feat = np.zeros((self.max_frame, feat_dim), dtype=np.float32)
        mask = np.ones((self.max_frame,), dtype=np.int32)
        if num_frames <= self.max_frame:
            feat[:num_frames] = raw_feats
            mask[num_frames:] = 0
        else:
            # if the number of frames exceeds the limitation, we need to sample
            # the frames.
            if self.test_mode:
                # uniformly sample when test mode is True
                step = num_frames // self.max_frame
                select_inds = list(range(0, num_frames, step))
                select_inds = select_inds[:self.max_frame]
            else:
                # randomly sample when test mode is False
                select_inds = list(range(num_frames))
                random.shuffle(select_inds)
                select_inds = select_inds[:self.max_frame]
                select_inds = sorted(select_inds)
            for i, j in enumerate(select_inds):
                feat[i] = raw_feats[j]
        feat = torch.FloatTensor(feat)
        mask = torch.LongTensor(mask)
        return feat, mask

    def tokenize_text(self, text: str) -> tuple:
        encoded_inputs = self.tokenizer(text, max_length=self.bert_seq_length, padding='max_length', truncation=True)
        input_ids = torch.LongTensor(encoded_inputs['input_ids'])
        mask = torch.LongTensor(encoded_inputs['attention_mask'])
        return input_ids, mask

    def __getitem__(self, idx: int) -> dict:
        # Step 1, load visual features from zipfile.
        frame_input, frame_mask = self.get_visual_feats(idx)

        # Step 2, load title tokens
        #加asr+ocr+title
        ocr_input=""
        maxlen1=self.bert_seq_length
        #print('DDD',self.anns[idx]['ocr'],'ffff')
        for a in self.anns[idx]["ocr"]:
            ocr_input+=a["text"]
        ocr_input1=self.tokenizer.tokenize(ocr_input)
        asr_input=self.anns[idx]["asr"]
        asr_input1=self.tokenizer.tokenize(asr_input)
        title_input=self.anns[idx]["title"]
        title_input1 = self.tokenizer.tokenize(title_input)

        if len(asr_input1)>int((maxlen1-4)/3):
            asr_input1=asr_input1[:int((maxlen1-4)/3)]


        if len(ocr_input1)>int((maxlen1-4)/3):
            ocr_input1=ocr_input1[:int((maxlen1-4)/3)]


        if len(title_input1)>int((maxlen1-4)/3):
            title_input1=title_input1[:int((maxlen1-4)/3)]
        text_input=["[CLS]"]+title_input1+["[SEP]"]+asr_input1+["[SEP]"]+ocr_input1+["[SEP]"]

        # for char in '''@$%^&*|()~`·、[]<>《》{}【】,.'‘:""''':
        #     ocr_input = ocr_input.replace(char, "")
        #     asr_input = self.anns[idx]["asr"].replace(char, "")
        #     title_input1= self.anns[idx]["title"].replace(char, "")
        #
        # if len(asr_input)+len(title_input1)+len(ocr_input)<maxlen1:
        #     input_text =title_input1+asr_input+ocr_input
        # elif len(title_input1)>int(maxlen1/2):
        #     oc = int((maxlen1+1)/6)
        #     if oc > len(ocr_input):
        #         oc = len(ocr_input)
        #     if int((maxlen1+1)/6)*2 > len(asr_input):
        #         t=maxlen1-oc-len(asr_input)
        #         input_text = title_input1[:t]+ asr_input+ ocr_input[:oc ]
        #     else:
        #         q = int((maxlen1+1)/6)
        #         h = maxlen1 - int(maxlen1/2) - oc - q
        #         input_text = title_input1[:int(maxlen1/2)] + asr_input[:q ] + asr_input[-h:] + ocr_input[:oc]
        #
        # else:
        #     oc=int((maxlen1-len(title_input1))/3)
        #     if oc > len(ocr_input):
        #         oc=len(ocr_input)
        #     if int((maxlen1-len(title_input1))/3)*2>len(asr_input):
        #         t=maxlen1-len(title_input1)-len(asr_input)
        #         input_text=title_input1+asr_input+ocr_input[:t]
        #     else:
        #         q=int((maxlen1-len(title_input1))/3)
        #         h=maxlen1-len(title_input1)-oc-q
        #         input_text=title_input1+asr_input[:q]+asr_input[-h:]+ocr_input[:oc]
        # if len(title_input1)>maxlen1:
        #     q=int(maxlen1/3)
        #     h=maxlen1-q
        #     title_input1=title_input1[:q]+title_input1[-h:]
        # if len(asr_input)>maxlen1:
        #     q = int(maxlen1 / 2)
        #     h = maxlen1 - q
        #     asr_input = asr_input[:q] + asr_input[-h:]
        # if len(ocr_input)>maxlen1:
        #     q = int(maxlen1 / 2)
        #     h = maxlen1 - q
        #     ocr_input = ocr_input[:q] + ocr_input[-h:]
        # words = self.tokenizer.tokenize(input_text)
        # # print('!!!!!!!!!!INFO:words:', words)
        # words = ["[CLS]"] + words
        # total_length_with_CLS = maxlen1 - 1
        # if len(words) > total_length_with_CLS:
        #     words = words[:total_length_with_CLS]
        # words = words + ["[SEP]"]
        #
        input_ids = self.tokenizer.convert_tokens_to_ids(text_input)
        seq_segment=[0]*(len(title_input1)+2)+[1]*(len(asr_input1)+1)+[2]*(len(ocr_input1)+1)
        # print(input_ids)
        # print(input_mask)
        input_mask = [1] * len(input_ids)
        # print(len(input_ids))
        while len(input_ids) < maxlen1:
            input_ids.append(0)
            input_mask.append(0)
            seq_segment.append(0)
        assert len(input_ids) == maxlen1
        assert len(input_mask) == maxlen1
        assert len(seq_segment) == maxlen1

        he_input = np.array(input_ids)
        he_mask = np.array(input_mask)
        he_segment = np.array(seq_segment)
        #title_input, title_mask = self.tokenize_text(self.anns[idx]['title'])
        # he_input, he_mask = self.tokenize_text(input_text)
        # title_input, title_mask = self.tokenize_text(title_input1)
        # asr_input, asr_mask = self.tokenize_text(asr_input)
        # ocr_input, ocr_mask = self.tokenize_text(ocr_input)

        # Step 3, summarize into a dictionary
        data = dict(
            frame_input=frame_input,
            frame_mask=frame_mask,
            # title_input=title_input,
            # title_mask=title_mask,
            he_input=he_input,
            he_mask=he_mask,
            he_segment=he_segment
            # ocr_input=ocr_input,
            # ocr_mask=ocr_mask,
            # asr_input=asr_input,
            # asr_mask=asr_mask
        )
        #print('##',data,'##')        


        # Step 4, load label if not test mode
        if not self.test_mode:
            label = category_id_to_lv2id(self.anns[idx]['category_id'])
            data['label'] = torch.LongTensor([label])

        return data
