import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import os
import h5py
import pprint
import pandas as pd 
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import os 
import re
import math
from tqdm import tqdm_notebook,tqdm
from sklearn.metrics import f1_score
from transformers import *

def ds_loader_label(filename):
  with open(filename, 'r') as fp:
    lines = [line.strip() for line in fp]   
  posts, post = [], []
  for line in tqdm_notebook(lines):
    probs=[]
    if line :
        annotations = line.split("\t")[2]
        # reading probabilities from the last column and also normalaize it by div on 9
        annotations = np.array(annotations.split('|'))
        probs.append(sum(annotations=='O'))
        probs.append(sum(annotations=='B'))
        probs.append(sum(annotations=='I'))
        probs = [probs[0],probs[2]+probs[1]]
        probs = [i/9 for i in probs]
        post.append(probs)
    elif post:
        posts.append(post)
        post = []
  # a list of lists of words/ labels
  return posts


def ds_loader_token(filename):
      with open(filename, 'r') as fp:
        lines = [line.strip() for line in fp]   
      posts, post = [], []
      ids,id =[],[]  
      for line in tqdm_notebook(lines):
          if line:
              words = line.split("\t")[1]
              word_id=line.split("\t")[0]  
              # print("words: ", words)
              post.append(words)
              id.append(word_id)  
          elif post:
              posts.append(post)
              ids.append(id)  
              id=[]  
              post = []
      # a list of lists of words/ labels
      if len(post):   
            posts.append(post)
            ids.append(id)  
      return posts,ids

train_tokens,train_ids=ds_loader_token('/home/pradyumna/emph_data/train.txt')
train_label=ds_loader_label('/home/pradyumna/emph_data/train.txt')

dev_tokens,dev_ids=ds_loader_token('/home/pradyumna/emph_data/dev.txt')
#dev_tokens,dev_ids=ds_loader_token('/media/nas_mount/Sarthak/sarthak/semeval/shuffle_dev/shuffle_dev_5.txt')
dev_label=ds_loader_label('/home/pradyumna/emph_data/dev.txt')

max_seq_length=max([len(token) for token in train_tokens])+2
max_seq_length

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None, label_mask=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask		

def convert_examples_to_features(tokens_set, labels_set, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    #label_map = {label: i for i, label in enumerate(label_list, 1)}

    input_ids, input_masks, segment_ids, labels = [], [], [], []
    for index in tqdm_notebook(range(len(tokens_set)),desc="Converting examples to features"):
        textlist = tokens_set[index] #example.text_a.split(' ')
        labellist = labels_set[index]
        input_id, input_mask, segment_id,label = convert_single_example(
            textlist, labellist,max_seq_length,tokenizer
        )
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
        labels.append(label)
    return (
        np.array(input_ids),
        np.array(input_masks),
        np.array(segment_ids),
        np.array(labels)
    )

def convert_single_example(textlist, labellist, max_seq_length,tokenizer):
  tokens = []
  labels = []
  for i, word in enumerate(textlist):
      token = tokenizer.tokenize(word)
      if token[0]=='▁' and len(token)>1:
        tokens.append(token[1])  
      else :  
        tokens.append(token[0])  
      labels.append(labellist[i])
  if len(tokens) >= max_seq_length - 1:
      tokens = tokens[0:(max_seq_length - 2)]
      labels = labels[0:(max_seq_length - 2)]
  ntokens = []
  segment_ids = []
  #ntokens.append("[CLS]")
  #segment_ids.append(0)
  labels.append([1,0])
  for i, token in enumerate(tokens):
      ntokens.append(token)
      segment_ids.append(0)
  #ntokens.append("[SEP]")
  segment_ids.append(0)  
  segment_ids.append(0)
  labels.append([1,0])
  input_ids = tokenizer.convert_tokens_to_ids(ntokens)
  input_ids.append(4)
  input_ids.append(3)  
  input_mask = [1] * len(input_ids)
  while len(input_ids) < max_seq_length:
      input_ids.insert(0,5)
      input_mask.insert(0,0)
      segment_ids.insert(0,3)
  while len(labels) < max_seq_length:
      labels.insert(0,[1,0])  
  assert len(labels) == max_seq_length    
  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length
  return input_ids,input_mask,segment_ids,labels
  
import tensorflow as tf
from transformers import XLNetTokenizer, TFXLNetModel,AlbertTokenizer,RobertaTokenizer

tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')

(train_input_ids, train_input_masks, train_segment_ids, train_labels)=convert_examples_to_features(train_tokens, train_label, max_seq_length, tokenizer)
(val_input_ids, val_input_masks, val_segment_ids, val_labels)=convert_examples_to_features(dev_tokens, dev_label, max_seq_length, tokenizer)

import torch
from transformers import *
import tensorflow as tf
#tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased',force_download=True)
model = XLNetModel.from_pretrained('xlnet-base-cased')#,force_download=True)

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.xlnet=XLNetModel.from_pretrained('xlnet-base-cased')
        #self.lstm=nn.LSTM(768, 128,
        #                  num_layers=1, bidirectional=True,batch_first=True)
        self.relu=nn.ReLU()
        #self.linear1=nn.Linear(768,512)
        self.linear2=nn.Linear(768,256)
        self.linear3=nn.Linear(256,64)
        self.linear=nn.Linear(64,2)
        self.softmax=nn.LogSoftmax(2)

    def forward(self, inputs):
        #inputs[0]=inputs[0].to(torch.device("cuda:1"))
        #inputs[1]=inputs[1].to(torch.device("cuda:1"))
        #inputs[2]=inputs[2].to(torch.device("cuda:1"))
        x = self.xlnet(input_ids=inputs[0],attention_mask=inputs[1],token_type_ids=inputs[2])[0]
        #x=x.to(torch.device("cuda:0"))
        #x,_ = self.lstm(x)
        #x=self.linear1(x)
        #x = self.relu(x)
        x=self.linear2(x)
        x = self.relu(x)
        x=self.linear3(x)
        x = self.relu(x)
        x = self.linear(x)
        x = self.softmax(x)
        return x

import torch.optim as optim
xlnet_model=Net()
model_clone = Net()

xlnet_model.cuda()
criterion = nn.KLDivLoss(reduction='batchmean')
optimizer = optim.Adam(xlnet_model.parameters(), lr=2e-5)

#val_inputs= [torch.tensor(val_input_ids),torch.tensor(val_input_masks),torch.tensor(val_segment_ids)]
bs=32
top=val_input_ids.shape[0]
max_score=0
#xlnet_model.cuda()
for epoch in range(30):  
    running_loss = 0.0
    for i in tqdm_notebook(range(math.ceil(train_input_ids.shape[0]/bs))):
        
        inputs= [torch.tensor(train_input_ids[i*bs:min(i*bs+bs,train_input_ids.shape[0])]).cuda(),
                 torch.tensor(train_input_masks[i*bs:min(i*bs+bs,train_input_ids.shape[0])]).cuda(),
                 torch.tensor(train_segment_ids[i*bs:min(i*bs+bs,train_input_ids.shape[0])]).cuda()]
        labels = torch.tensor(train_labels[i*bs:i*bs+bs]).cuda()

        
        optimizer.zero_grad()

        
        outputs = xlnet_model(inputs)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()

        
        running_loss += loss.item()
        #print("loss ",running_loss/(i+1))

        #print(f1_score(np.reshape(np.argmax(labels,axis=2),[labels.shape[0],labels.shape[1]]), 
        #                np.reshape(np.argmax(outputs.detach().numpy(),axis=2),[labels.shape[0],labels.shape[1]]),average='micro'))
        #print('-'*10)   
    xlnet_model.to(torch.device("cpu:0"))
    predictions=xlnet_model([torch.tensor(val_input_ids[0][np.newaxis,:]),torch.tensor(val_input_masks[0][np.newaxis,:]),torch.tensor(val_segment_ids[0][np.newaxis,:])])
    for j in tqdm(range(math.ceil(val_input_ids.shape[0]/bs))):
      predict=xlnet_model([torch.tensor(val_input_ids[j*bs:min(j*bs+bs,top)]),torch.tensor(val_input_masks[j*bs:min(j*bs+bs,top)]),torch.tensor(val_segment_ids[j*bs:min(j*bs+bs,top)])])
      predictions=torch.cat((predictions,predict),0)
    predictions=predictions[1:]    
    score=scorer(torch.exp(predictions),dev_tokens)
    torch.save(xlnet_model.state_dict(), "/media/data_dump/Pradyumna/empha/pyt-D2xlnet-{}".format(epoch))
    if score>max_score:
        model_clone.load_state_dict(xlnet_model.state_dict())
        max_score=score
    xlnet_model.cuda()
    #print(f1_score(np.reshape(np.argmax(labels,axis=2),[labels.shape[0],labels.shape[1]]), 
    #                    np.reshape(np.argmax(predictions.detach().numpy(),axis=2),[labels.shape[0],labels.shape[1]]),average='micro'))
    #torch.save(xlnet_model.state_dict(), "/media/data_dump/Pradyumna/empha/pyt-temp-{}-{}.pt".format(epoch,2e-5))

    print('-'*10)


        #if i % 2000 == 1999:    # print every 2000 mini-batches
        #    print('[%d, %5d] loss: %.3f' %
        #          (epoch + 1, i + 1, running_loss / 2000))
        #    running_loss = 0.0

print('Finished Training')
		

