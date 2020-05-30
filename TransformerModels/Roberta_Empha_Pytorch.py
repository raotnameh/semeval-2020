import torch
from transformers import *
import tensorflow as tf
#tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased',force_download=True)
model = RobertaModel.from_pretrained('roberta-base')#,force_download=True)

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.roberta=model
        self.lstm=nn.LSTM(768, 128,
                          num_layers=1, bidirectional=True,batch_first=True)
        self.relu=nn.ReLU()
        self.linear=nn.Linear(256,2)
        self.softmax=nn.LogSoftmax(2)

    def forward(self, inputs):
        #inputs[0]=inputs[0].to(torch.device("cuda:1"))
        #inputs[1]=inputs[1].to(torch.device("cuda:1"))
        #inputs[2]=inputs[2].to(torch.device("cuda:1"))
        x = self.roberta(input_ids=inputs[0],attention_mask=inputs[1],token_type_ids=inputs[2])[0]
        #x=x.to(torch.device("cuda:0"))
        x,_ = self.lstm(x)
        x = self.relu(x)
        x = self.linear(x)
        x = self.softmax(x)
        return x
		

import torch.optim as optim
roberta_model=Net()
model_clone = Net()

####Block To Freeze Some Layers of the Roberta Model



#for name, param in XLNetModel.from_pretrained('xlnet-base-cased').named_parameters():                
#    if param.requires_grad:
#        print(name)




for name, param in roberta_model.named_parameters():                
    if "roberta" in name and "11" not in name:
        param.requires_grad=False
    if "roberta" in name and "11" in name:
        param.requires_grad=True

for name, param in roberta_model.named_parameters():                
    if param.requires_grad:
        print(name)

roberta_model.cuda()
criterion = nn.KLDivLoss(reduction='batchmean')
optimizer = optim.Adam(roberta_model.parameters(), lr=9e-5)

#val_inputs= [torch.tensor(val_input_ids),torch.tensor(val_input_masks),torch.tensor(val_segment_ids)]
bs=32
top=val_input_ids.shape[0]
max_score=0
for epoch in range(20):  
    running_loss = 0.0
    for i in tqdm_notebook(range(math.ceil(train_input_ids.shape[0]/bs))):
        # get the inputs; data is a list of [inputs, labels]
        inputs= [torch.tensor(train_input_ids[i*bs:min(i*bs+bs,train_input_ids.shape[0])]).cuda(),
                 torch.tensor(train_input_masks[i*bs:min(i*bs+bs,train_input_ids.shape[0])]).cuda(),
                 torch.tensor(train_segment_ids[i*bs:min(i*bs+bs,train_input_ids.shape[0])]).cuda()]
        labels = torch.tensor(train_labels[i*bs:i*bs+bs]).cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = roberta_model(inputs)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        #print("loss ",running_loss/(i+1))

        #print(f1_score(np.reshape(np.argmax(labels,axis=2),[labels.shape[0],labels.shape[1]]), 
        #                np.reshape(np.argmax(outputs.detach().numpy(),axis=2),[labels.shape[0],labels.shape[1]]),average='micro'))
        #print('-'*10)   
    roberta_model.to(torch.device("cpu:0"))
    torch.save(roberta_model.state_dict(), "/media/data_dump/Pradyumna/empha/L_R-{}".format(epoch))
    predictions=roberta_model([torch.tensor(val_input_ids[0][np.newaxis,:]),torch.tensor(val_input_masks[0][np.newaxis,:]),torch.tensor(val_segment_ids[0][np.newaxis,:])])
    for j in tqdm(range(math.ceil(val_input_ids.shape[0]/bs))):
      predict=roberta_model([torch.tensor(val_input_ids[j*bs:min(j*bs+bs,top)]),torch.tensor(val_input_masks[j*bs:min(j*bs+bs,top)]),torch.tensor(val_segment_ids[j*bs:min(j*bs+bs,top)])])
      predictions=torch.cat((predictions,predict),0)
    predictions=predictions[1:]    
    score=scorer(torch.exp(predictions),dev_tokens)
    if score>max_score:
        model_clone.load_state_dict(roberta_model.state_dict())
        max_score=score
    roberta_model.cuda()
    #print(f1_score(np.reshape(np.argmax(labels,axis=2),[labels.shape[0],labels.shape[1]]), 
    #                    np.reshape(np.argmax(predictions.detach().numpy(),axis=2),[labels.shape[0],labels.shape[1]]),average='micro'))
    #torch.save(xlnet_model.state_dict(), "/media/data_dump/Pradyumna/empha/pyt-temp-{}-{}.pt".format(epoch,2e-5))

    print('-'*10)


        #if i % 2000 == 1999:    # print every 2000 mini-batches
        #    print('[%d, %5d] loss: %.3f' %
        #          (epoch + 1, i + 1, running_loss / 2000))
        #    running_loss = 0.0

print('Finished Training')		

roberta_model.load_state_dict(torch.load("/media/data_dump/Pradyumna/empha/L_R-5"))
