import numpy as np
import pandas as pd
import tqdm
import sys
import pickle
import gensim
import os
from collections import OrderedDict
def read_stopwords(infile):
    res = {}
    for line in open(infile,encoding='utf-8'):
        line = line.strip()
        res[line] = 1
    return res


def read_input(infile, stopwords):
    lineid = 0
    field_num = 0
    datas = []
    docid = 0
    datasdict = {'text_right':[],'text_left':[],'label':[]}
    for line in open(infile,encoding='utf-8'):
        lineid += 1
        #if(lineid==100):
         #   break
        if lineid == 1:
            field_num = len(line.split('|'))
            continue
        line = line.strip()
        fields = line.split('|')
        if len(fields) != field_num:
            sys.stderr.write('format error1\t' + line + '\n')
            continue
        cur_data = {}
        cur_data['label'] = int(fields[0])

        cur_data['doc1'] = {}
        cur_data['doc1']['orig'] = fields[5].strip()
        cur_data['doc1']['text'] = cur_data['doc1']['orig'].replace(' ', '')
        cur_data['doc1']['tokens'] = cur_data['doc1']['orig'].split(' ')
        cur_data['doc1']['tokens_without_stopwords'] = [w for w in cur_data['doc1']['tokens'] if w not in stopwords]
        cur_data['doc1']['docid'] = docid
        docid += 1

        cur_data['doc2'] = {}
        cur_data['doc2']['orig'] = fields[6].strip()
        cur_data['doc2']['text'] = cur_data['doc2']['orig'].replace(' ', '')
        cur_data['doc2']['tokens'] = cur_data['doc2']['orig'].split(' ')
        cur_data['doc2']['tokens_without_stopwords'] = [w for w in cur_data['doc2']['tokens'] if w not in stopwords]
        cur_data['doc2']['docid'] = docid
        docid += 1
        datas.append(cur_data)
        datasdict['label'].append(int(fields[0]))
        datasdict['text_left'].append(cur_data['doc1']['text'])
        datasdict['text_right'].append(cur_data['doc2']['text'])
    return datas,datasdict


MAX_SEQUENCE_LENGTH = 350
#input_categories = ['q1','q2']
output_categories = 'label'
#event_file = '/Users/wit/OneDrive - mail.scut.edu.cn/njusearch/data/news pairs/raw/event-story-cluster/same_event_doc_pair.txt'
#print('-------------')
#stopwords_file = '/Users/wit/OneDrive - mail.scut.edu.cn/njusearch/data/news pairs/raw/event-story-cluster/stopwords-zh.txt'
#print('-------------')
event_file = r"/home/data_ti4_c/zongwz/data/newsPairs/raw/event-story-cluster/same_event_doc_pair.txt"

#print('-------------')
stopwords_file =  r'home/data_ti4_c/zongwz/data/newsPairs/raw/event-story-cluster/stopwords-zh.txt'
#print('-------------')
d,datas = read_input(event_file,stopwords_file)
#print('-------------')
Datas = pd.DataFrame(datas)
Datas['label']=Datas['label'].astype(int)
input_categories = ['text_left','text_right']

import torch

from torch.utils.data import Dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def _convert_to_transformer_inputs(question, answer, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for transformer (including bert)"""
    
    def return_id(str1, str2, truncation_strategy, length):

        inputs = tokenizer.encode_plus(str1, str2,
            add_special_tokens=True,
            max_length=length,
            truncation_strategy=truncation_strategy,
            truncation=True
            )
        
        input_ids =  inputs["input_ids"]
        #print(tokenizer.decode(input_ids))
        input_masks = [1] * len(input_ids)
        input_segments = inputs["token_type_ids"]
        padding_length = length - len(input_ids)
        padding_id = tokenizer.pad_token_id
        input_ids = input_ids + ([padding_id] * padding_length)
        input_masks = input_masks + ([0] * padding_length)
        input_segments = input_segments + ([0] * padding_length)
        
        return [input_ids, input_masks, input_segments]
    
    input_ids_q, input_masks_q, input_segments_q = return_id(
        question, answer, 'longest_first', max_sequence_length)
    

    
    return [input_ids_q, input_masks_q, input_segments_q]

def compute_input_arrays(df, columns, tokenizer, max_sequence_length):
    input_ids_q, input_masks_q, input_segments_q = [], [], []
    input_ids_a, input_masks_a, input_segments_a = [], [], []
    for _, instance in tqdm(df[columns].iterrows()):
        q, a = instance.text_left, instance.text_right

        ids_q, masks_q, segments_q= \
        _convert_to_transformer_inputs(q, a, tokenizer, max_sequence_length)
        
        input_ids_q.append(ids_q)
        input_masks_q.append(masks_q)
        input_segments_q.append(segments_q)

    return [np.asarray(input_ids_q, dtype=np.int32), 
            np.asarray(input_masks_q, dtype=np.int32), 
            np.asarray(input_segments_q, dtype=np.int32)]

def compute_output_arrays(df, columns):
    return np.asarray(df[columns])


def search_f1(y_true, y_pred):
    best = 0
    best_t = 0
    for i in range(30,60):
        tres = i / 100
        y_pred_bin =  (y_pred > tres).astype(int)
        score = f1_score(y_true, y_pred_bin)
        if score > best:
            best = score
            best_t = tres
    print('best', best)
    print('thres', best_t)
    return best, best_t

from tqdm import tqdm
import transformers
from transformers import BertTokenizer, BertForNextSentencePrediction,BertModel
from transformers import AdamW
import pickle
from torch.utils.data import DataLoader
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
bertmodel = BertModel.from_pretrained('bert-base-chinese', return_dict=True)
#bertmodel.cuda()


outputs = compute_output_arrays(Datas, output_categories)
inputs = compute_input_arrays(Datas, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)

class Metrics(object):
  '''
    包括f1score、acc、recall的计算方法
  '''
  
  def __init__(self):
    self._tot_num = 0.0
    self._true_num = 0.0
    self._postive_num = 0.0
    self._true_postive_num = 0.0
    self._acc = 0.0
    self._recall = 0.0
    self._f1 = 0.0
    self._loss = 0.0

  def add_arg(self,Totnum = 0.0,truenum=0.0,postivenum=0.0,truepostivenum=0.0,Loss=0.0):
    self._tot_num+=Totnum
    self._postive_num+=postivenum
    self._true_num+=truenum
    self._true_postive_num+=truepostivenum
    self._loss+=Loss

  def compute(self):
    self._acc = self._true_num/self._tot_num
    self._recall = self._true_postive_num/self._true_num
    self._f1 = 2 * self._acc * self._recall / (self._acc + self._recall)
    self._loss = self._loss/self._tot_num
    return self._acc,self._recall,self._f1,self._loss
    

from sklearn.model_selection import KFold
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
epoch_num = 10
gkf = KFold(5)
#dataset = Mydata(input_ids,token_type_ids,attention_mask)
batch_size = 1
#dataloader = DataLoader(dataset, batch_size = 3, shuffle=True)

class BertMatch(nn.Module):
    def __init__(self):
        super(BertMatch,self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese', return_dict=True)
        self.cls = nn.Linear(768,2)

    def forward(self,input_ids,attention_mask,token_type_ids):
        bertout = self.bert(input_ids,attention_mask,token_type_ids)['pooler_output']
        bertout = self.cls(bertout)
        #[batchsize,seq_size,hid_size]
        bertout = F.sigmoid(bertout)
        out = F.softmax(bertout,dim = 0)
        #[batch_size,2]
        return out
model = BertMatch()
model = model.to(device)
learning_rate = 3e-7
optim = torch.optim.AdamW(model.parameters(),lr = learning_rate)
criterion = nn.MSELoss(reduce=True,size_average=True)
for train_idx, valid_idx in gkf.split(list(range(len(inputs[0])))) :
    train_inputs = [inputs[i][train_idx] for i in range(len(inputs))]
    print(inputs)
    print(len(inputs))
    print(type(outputs))
    train_outputs = outputs[train_idx]
    
    
    valid_inputs = [inputs[i][valid_idx] for i in range(len(inputs))]
    valid_outputs = outputs[valid_idx]
    cnt = 0
    print(len(train_inputs[0]))
    for epoch in range(epoch_num):
      train_metric = Metrics()
      valid_metric = Metrics()
      for cnt in tqdm(range(0,len(train_inputs[0]),batch_size)):
        train_output=None
        if(cnt+batch_size<len(train_inputs[0])):
          idx = list(range(cnt,cnt+batch_size))
          train_input = [train_inputs[i][idx] for i in range(3)]
          train_output = [0.0 if i != train_outputs[idx] else 1.0 for i in range(2)]
        else:
          idx = list(range(cnt,len(train_inputs[0])))
          train_input = [train_inputs[i][idx] for i in range(3)]
          train_output = [0.0 if i != train_outputs[idx] else 1.0 for i in range(2)]
        train_input = torch.LongTensor(train_input)
        train_output = torch.LongTensor(train_output)
        train_input = train_input.to(device)
        #train_output = train_output.cuda()
        train_output = train_output.squeeze(-1).to(device)
        optim.zero_grad()
        model.train()
        if hasattr(torch.cuda, 'empty_cache'):
          torch.cuda.empty_cache()
        #print(train_input.size())
        #print(train_output.size())
        #print(train_input[0].size())
        
        output = model(input_ids=train_input[0], attention_mask=train_input[1], token_type_ids=train_input[2])
        loss = criterion(output.cuda(),train_output)
        print(output)
        pred = torch.argmax(output,dim = -1)
        print(pred)
        target = torch.argmax(train_output,dim = -1)
        true_num = float(torch.sum(pred == target))
        tot_num = float(batch_size)
        postive_num = float(torch.sum(target==1))
        true_postive_num = float(torch.sum(torch.logical_and(target==1,pred)))
        train_metric.add_arg( 1.0,true_num,postive_num,true_postive_num,loss)
        
        #print(outputs)
        loss.backward()
        optimizer.step()
        
      for id in range(len(valid_inputs[0])):
        valid_input = [[valid_inputs[i][id]] for i in range(3)]
        valid_output = [0.0 if i != valid_outputs[idx] else 1.0 for i in range(2)]
        valid_output = torch.LongTensor(valid_output).to(device)
        valid_input = torch.LongTensor(valid_input).to(device)
        model.eval()
        if hasattr(torch.cuda, 'empty_cache'):
          torch.cuda.empty_cache()
        output = model(input_ids=valid_input[0], attention_mask=valid_input[1], token_type_ids=valid_input[2])
        loss = criterion(output,target)
        model.train()
        #print(output['logits'])
        
        pred = torch.argmax(output,dim = -1)
        target = torch.argmax(valid_output,dim=-1)
        true_num = float(torch.sum(pred == target))
        tot_num = float(1.0)
        postive_num = float(torch.sum(target==1))
        true_postive_num = float(torch.sum(torch.logical_and(target==1,pred)))
        valid_metric.add_arg( 1.0,true_num,postive_num,true_postive_num,loss)
      valid_acc,valid_recall,valid_f1,valid_loss = valid_metric.compute()
      train_acc,train_recall,train_f1,train_loss = train_metric.compute()
      print('train:acc:%s f1:%s recall:%s loss:%s' % (train_acc,train_f1,train_recall,valid_loss))
      print('valid:acc:%s f1:%s recall:%s loss:%s' % (valid_acc,valid_f1,valid_recall,train_loss))