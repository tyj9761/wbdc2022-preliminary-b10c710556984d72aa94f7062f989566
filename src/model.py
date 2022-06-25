import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

from category_id_map import CATEGORY_ID_LIST


class MultiModal(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.bert = BertModel.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
        self.video_fc = torch.nn.Linear(768, 768)
        self.classifier = nn.Linear(args.fc_size, len(CATEGORY_ID_LIST))

    def forward(self, inputs, inference=False,inference1=False):
        text_embedding = self.bert.embeddings(inputs['title_input'])
        # text input is [CLS][SEP] t e x t [SEP]
        cls_embedding = text_embedding[:, 0:1, :]
        text_embedding = text_embedding[:, 1:, :]
        cls_mask = inputs['title_mask'][:, 0:1]
        text_mask = inputs['title_mask'][:, 1:]

        video_feature = self.video_fc(inputs['frame_input'])
        video_embedding = self.bert.embeddings(inputs_embeds=video_feature)

        # [CLS] Video [SEP] Text [SEP]
        embedding_output = torch.cat([cls_embedding, video_embedding, text_embedding], 1)
        mask = torch.cat([cls_mask,inputs['frame_mask'],text_mask], 1)

        mask = mask[:, None, None, :]
        mask = (1.0 - mask) * -10000.0

        embedding_output = self.bert.encoder(embedding_output, attention_mask=mask)['last_hidden_state']

        mean_pooling_embeddings = torch.mean(embedding_output, 1)

        prediction = self.classifier(mean_pooling_embeddings)

        if inference:
            return torch.argmax(prediction, dim=1)
        elif inference1:
            return prediction
        else:
            return self.cal_loss(prediction, inputs['label'])

    @staticmethod
    def cal_loss(prediction, label):
        label = label.squeeze(dim=1)
        loss = F.cross_entropy(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label


