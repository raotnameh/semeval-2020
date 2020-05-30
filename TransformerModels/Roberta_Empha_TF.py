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
        
def create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""
    bert_module =  hub.Module(bert_path)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    vocab_file, do_lower_case = sess.run(
        [
            tokenization_info["vocab_file"],
            tokenization_info["do_lower_case"],
        ]
    )

    return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

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
      #token = tokenizer.tokenize(word)
      if i>0 :
        token = tokenizer.tokenize("i "+word)  
        tokens.append(token[1])
      else :
        token = tokenizer.tokenize(word)
        tokens.append(token[0])
      labels.append(labellist[i])
  if len(tokens) >= max_seq_length - 1:
      tokens = tokens[0:(max_seq_length - 2)]
      labels = labels[0:(max_seq_length - 2)]
  ntokens = []
  segment_ids = []
  #ntokens.append("[CLS]")
  segment_ids.append(0)
  labels.insert(0,[1,0])
  for i, token in enumerate(tokens):
      ntokens.append(token)
      segment_ids.append(0)
  #ntokens.append("[SEP]")
  segment_ids.append(0)
  labels.append([1,0])
  input_ids = tokenizer.convert_tokens_to_ids(ntokens)
  input_ids.insert(0,0)
  input_ids.append(2)  
  input_mask = [1] * len(input_ids)
  while len(input_ids) < max_seq_length:
      input_ids.append(1)
      input_mask.append(0)
      segment_ids.append(0)
  while len(labels) < max_seq_length:
      labels.append([1,0])  
  assert len(labels) == max_seq_length    
  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length
  return input_ids,input_mask,segment_ids,labels
  
import tensorflow as tf
from transformers import XLNetTokenizer, TFXLNetModel,AlbertTokenizer,RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

(train_input_ids, train_input_masks, train_segment_ids, train_labels)=convert_examples_to_features(train_tokens, train_label, max_seq_length, tokenizer)
(val_input_ids, val_input_masks, val_segment_ids, val_labels)=convert_examples_to_features(dev_tokens, dev_label, max_seq_length, tokenizer)

from tensorflow.keras import backend as K

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    true=K.cast(K.flatten(K.argmax(y_true,axis=2)),dtype='float32')
    pred=K.cast(K.flatten(K.argmax(y_pred,axis=2)),dtype='float32')
    precision = precision(true, pred)
    recall = recall(true, pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

import numpy as np
import sys
import os
import os.path


def average(lst):
    return sum(lst) / float(len(lst))


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


def match_m(all_scores, all_labels):
    """
    This function computes match_m.
    :param all_scores: submission scores
    :param all_labels: ground_truth labels
    :return: match_m dict
    """
    print("[LOG] computing Match_m . . .")
    top_m = [1, 2, 3, 4]
    match_ms = {}
    for m in top_m:
        print("[LOG] computing m={} in match_m".format(m))
        intersects_lst = []
        # ****************** computing scores:
        score_lst = []
        for s in all_scores:
            # the length of sentence needs to be more than m:
            if len(s) <= m:
                continue
            s = np.array(s)
            ind_score = np.argsort(s)[-m:]
            score_lst.append(ind_score.tolist())
        # ****************** computing labels:
        label_lst = []
        for l in all_labels:
            # the length of sentence needs to be more than m:
            if len(l) <= m:
                continue
            # if label list contains several top values with the same amount we consider them all
            h = m
            if len(l) > h:
                while (l[np.argsort(l)[-h]] == l[np.argsort(l)[-(h + 1)]] and h < (len(l) - 1)):
                    h += 1
            l = np.array(l)
            ind_label = np.argsort(l)[-h:]
            label_lst.append(ind_label.tolist())

        for i in range(len(score_lst)):
            # computing the intersection between scores and ground_truth labels:
            intersect = intersection(score_lst[i], label_lst[i])
            intersects_lst.append((len(intersect))/float((min(m, len(score_lst[i])))))
        # taking average of intersects for the current m:
        match_ms[m] = average(intersects_lst)

    return match_ms


def read_results(filename):
    lines = read_lines(filename) + ['']
    e_freq_lst, e_freq_lsts = [], []

    for line in lines:
        if line:
            splitted = line.split("\t")
            e_freq = splitted[2]
            e_freq_lst.append(e_freq)

        elif e_freq_lst:
            e_freq_lsts.append(e_freq_lst)
            e_freq_lst = []
    return e_freq_lsts


def read_labels(filename):
    lines = read_lines(filename) + ['']
    e_freq_lst, e_freq_lsts = [], []

    for line in lines:
        if line:
            splitted = line.split("\t")
            e_freq = splitted[4]
            e_freq_lst.append(e_freq)

        elif e_freq_lst:
            e_freq_lsts.append(e_freq_lst)
            e_freq_lst = []
    return e_freq_lsts


def read_lines(filename):
    with open(filename, 'r') as fp:
        lines = [line.strip() for line in fp]
    return lines


def scorer(predictions,dev_tokens,dev_ids):
    with open("/home/pradyumna/emph_data/prediction.txt",'w') as f:
        for i,example in tqdm_notebook(enumerate(dev_tokens)):
            for j,token in enumerate(example):
                f.write(dev_ids[i][j]+'\t'+token+'\t'+str(predictions[i][j+1][1])+'\t'+'\n')
            f.write('\n')    
        
    all_score = read_results("/home/pradyumna/emph_data/prediction.txt")
    all_label = read_labels("/home/pradyumna/emph_data/dev.txt")

    assert len(all_score) == len(all_label)
    for i in range(len(all_label)):
        assert len(all_label[i]) == len(all_score[i])

    matchm = match_m(all_score, all_label)
    print("[LOG] Match_m: ", matchm)
    print("[LOG] computing RANKING score")

    sum_of_all_scores = 0
    for key,value in matchm.items():
        #output_file.write("score"+str(key)+":"+str(value))
        #output_file.write("\n")
        sum_of_all_scores+=value
    print("score:"+str(sum_of_all_scores/float(4))+"\n") #score for final "computed score"
    
    return sum_of_all_scores/float(4)

from tensorflow.keras.layers import Input,Dense,Bidirectional,LSTM

for lr in [2e-5]:
    for epochs in [3,5,7,10]:
        for layer in [3]:#[1,2,3]:
            for dropout in [0]:#,0.1,0.3]:
                tf.keras.backend.clear_session()
                
                token_inputs = tf.keras.layers.Input(shape=(None,), name='word_inputs', dtype=tf.int32)
                mask_inputs = tf.keras.layers.Input(shape=(None,), name='mask_inputs', dtype=tf.int32)
                seg_inputs = tf.keras.layers.Input(shape=(None,), name='seg_inputs', dtype=tf.int32)

                inputs=[token_inputs,mask_inputs,seg_inputs]

                transformer_outputs= TFRobertaModel.from_pretrained('roberta-base')(inputs)[0]

                #albert_outputs=K.squeeze(albert_last_outputs[:, 0:1, :], axis=1)

                step=transformer_outputs
                
                #step=tf.keras.layers.BatchNormalization()(step)
                if layer>=3:
                  step=tf.keras.layers.Dense(512,activation='relu')(step)
                  if dropout!=0:
                      step=tf.keras.layers.Dropout(rate=dropout)(step)
                if layer>=2:
                  step=tf.keras.layers.Dense(256,activation='relu')(step)
                  if dropout!=0:
                      step=tf.keras.layers.Dropout(rate=dropout)(step)
                if layer>=1:    
                  step=tf.keras.layers.Dense(64,activation='relu')(step)
                  if dropout!=0:
                      step=tf.keras.layers.Dropout(rate=dropout)(step)    
                #step=tf.keras.layers.Dropout(rate=dropout)(step)            
                
                #step=Bidirectional(LSTM(128,return_sequences=True,activation='relu'))(step)


                pred=tf.keras.layers.Dense(2,activation='softmax')(step)

                model=tf.keras.Model(inputs=inputs,outputs=pred)

                #model.layers[3].trainable=False


                model.compile(loss='kullback_leibler_divergence',
                      optimizer=tf.keras.optimizers.Adam(lr=lr),
                      metrics=[f1,'accuracy'])
                model.summary()

                #from sklearn.utils.class_weight import compute_class_weight
                #class_weights = compute_class_weight('balanced', np.unique(data_train.iloc[:,1]), data_train.iloc[:,1])

                model.fit([train_input_ids,train_input_masks,train_segment_ids],
                      train_labels,
                      epochs=epochs,
                      batch_size=32,
                      validation_data=([val_input_ids,val_input_masks,val_segment_ids],val_labels),
                      #class_weight=dict(enumerate(class_weights))  
                      #callbacks=[
                          #tf.keras.caltlbacks.LearningRateScheduler(scheduler,verbose=1)
                          #TensorBoardColabCallback(tbc)
                      #  ] 
                        )
                predictions=model.predict([val_input_ids, val_input_masks, val_segment_ids])
                scorer(predictions,dev_tokens,dev_ids)  
                print(f1_score(np.reshape(np.argmax(val_labels,axis=2),[val_labels.shape[0],val_labels.shape[1]]), 
                            np.reshape(np.argmax(predictions,axis=2),[val_labels.shape[0],val_labels.shape[1]]),average='micro'))
                print(f1_score(np.reshape(np.argmax(val_labels,axis=2),[val_labels.shape[0],val_labels.shape[1]]), 
                            np.reshape(np.argmax(predictions,axis=2),[val_labels.shape[0],val_labels.shape[1]]),average='macro'))          
                #model.save('/media/data_dump/Pradyumna/empha/model-{}-{}-{}-{}-12.h5'.format(lr,epochs,dropout,layers))
                #model_json = model.to_json()
                #with open('/media/data_dump/Pradyumna/empha/roberta-{}-{}-{}-{}.json'.format(lr,epochs), "w") as json_file:
                #    json_file.write(model_json)
                # serialize weights to HDF5
                model.save_weights('/media/data_dump/Pradyumna/empha/Droberta-{}-{}-{}-{}.h5'.format(lr,epochs,layer,dropout))  
                print("Done!!")	
				
				