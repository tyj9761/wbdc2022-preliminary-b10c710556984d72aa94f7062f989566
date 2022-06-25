import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from transformers import BertModel
from transformers.models.bert.modeling_bert import BertEmbeddings,BertConfig
from category_id_map import CATEGORY_ID_LIST


class MultiModal(nn.Module):
    def __init__(self, args, task=['mlm', 'mfm'], init_from_pretrain=True):
        super().__init__()

        config=BertConfig.from_pretrained(args.bert_dir)
        # self.nextvlad_diff = NeXtVLAD(args.frame_embedding_size, args.vlad_cluster_size,
        #                               output_size=args.vlad_hidden_size, dropout=args.dropout, name='nextVLADdiff')
        self.bert = BertModel.from_pretrained(args.bert_dir, cache_dir=args.bert_cache,config=config)
        self.nextvlad = NeXtVLAD(args.frame_embedding_size, args.vlad_cluster_size,
                                 output_size=args.vlad_hidden_size, dropout=args.dropout)
        self.enhance = SENet(channels=args.vlad_hidden_size, ratio=args.se_ratio)
        self.enhance1 = SENet(channels=1792, ratio=args.se_ratio)
        bert_output_size = 768
        self.dropout1 = torch.nn.Dropout(0.2)
        self.classifier = nn.Linear(768, len(CATEGORY_ID_LIST))
        self.classifier1 = nn.Linear(768, 768)
        
    
    def forward(self, inputs, inference=False,inference1=False):
        v=torch.relu(self.classifier1(inputs['frame_input']))
        he_embedding=self.bert.embeddings(inputs['he_input'])
        v_embedding = self.bert.embeddings(inputs_embeds=v)

        f_embedding=torch.cat((he_embedding,v_embedding),dim=1)
        f_mask=torch.cat((inputs['he_mask'],inputs['frame_mask']),dim=1)

        f_masks=f_mask[:,None,None,:]
        f_masks=(1-f_masks)*-10000.0
        encoder_outputs=self.bert.encoder(f_embedding,attention_mask=f_masks,return_dict=True,output_hidden_states=False)
        last_embedding=encoder_outputs['last_hidden_state'].transpose(1, 2)
        
        last_mean=torch.avg_pool1d(last_embedding, kernel_size=last_embedding.shape[-1]).squeeze(-1)
        prediction = self.classifier(last_mean)

        if inference:
            return torch.argmax(prediction, dim=1)
            #return prediction
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


class NeXtVLAD(nn.Module):
    def __init__(self, feature_size, cluster_size, output_size=1024, expansion=2, groups=8, dropout=0.2,name='nextVLAD'):
        super().__init__()
        self.feature_size = feature_size
        self.output_size = output_size
        self.expansion_size = expansion
        self.cluster_size = cluster_size
        self.groups = groups
        self.drop_rate = dropout

        self.new_feature_size = self.expansion_size * self.feature_size // self.groups

        self.dropout = torch.nn.Dropout(self.drop_rate)
        self.expansion_linear = torch.nn.Linear(self.feature_size, self.expansion_size * self.feature_size)
        self.group_attention = torch.nn.Linear(self.expansion_size * self.feature_size, self.groups)
        self.cluster_linear = torch.nn.Linear(self.expansion_size * self.feature_size, self.groups * self.cluster_size,
                                              bias=False)
        self.cluster_weight = torch.nn.Parameter(
            torch.nn.init.normal_(torch.rand(1, self.new_feature_size, self.cluster_size), std=0.01))
        self.fc = torch.nn.Linear(self.new_feature_size * self.cluster_size, self.output_size)

    def forward(self, inputs, mask):
        # todo mask
        inputs = self.expansion_linear(inputs)
        attention = self.group_attention(inputs)
        attention = torch.sigmoid(attention)
        attention = attention.reshape([-1, inputs.size(1) * self.groups, 1])
        reshaped_input = inputs.reshape([-1, self.expansion_size * self.feature_size])
        activation = self.cluster_linear(reshaped_input)
        activation = activation.reshape([-1, inputs.size(1) * self.groups, self.cluster_size])
        activation = torch.softmax(activation, dim=-1)
        activation = activation * attention
        a_sum = activation.sum(-2, keepdim=True)
        a = a_sum * self.cluster_weight
        activation = activation.permute(0, 2, 1).contiguous()
        reshaped_input = inputs.reshape([-1, inputs.shape[1] * self.groups, self.new_feature_size])
        vlad = torch.matmul(activation, reshaped_input)
        vlad = vlad.permute(0, 2, 1).contiguous()
        vlad = F.normalize(vlad - a, p=2, dim=1)
        vlad = vlad.reshape([-1, self.cluster_size * self.new_feature_size])
        vlad = self.dropout(vlad)
        vlad = self.fc(vlad)
        return vlad


class SENet(nn.Module):
    def __init__(self, channels, ratio=8):
        super().__init__()
        self.sequeeze = nn.Linear(in_features=channels, out_features=channels // ratio, bias=False)
        self.relu = nn.ReLU()
        self.excitation = nn.Linear(in_features=channels // ratio, out_features=channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gates = self.sequeeze(x)
        gates = self.relu(gates)
        gates = self.excitation(gates)
        gates = self.sigmoid(gates)
        x = torch.mul(x, gates)

        return x


class ConcatDenseSE(nn.Module):
    def __init__(self, multimodal_hidden_size, hidden_size, se_ratio, dropout):
        super().__init__()
        self.fusion = nn.Linear(multimodal_hidden_size, hidden_size)
        self.fusion_dropout = nn.Dropout(dropout)
        self.enhance = SENet(channels=hidden_size, ratio=se_ratio)

    def forward(self, inputs):
        embeddings = torch.cat(inputs, dim=1)
        embeddings = self.fusion_dropout(embeddings)
        embedding = self.fusion(embeddings)
        embedding = self.enhance(embedding)

        return embedding


class SoftAttention(nn.Module): 
    def __init__(self, hidden_size):
        super(SoftAttention, self).__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def get_attn(self, reps, mask=None):
        reps = torch.unsqueeze(reps, 1)
        attn_socres = self.attn(reps).squeeze(2)
        if mask is not None:
            attn_socres = mask * attn_socres
        attn_weight = attn_socres.unsqueeze(2)
        attn_out = torch.sum(reps * attn_weight, dim=1)

        return attn_out
    
    def forward(self, reps, mask=None):
        attn_out = self.get_attn(reps, mask)

        return attn_out