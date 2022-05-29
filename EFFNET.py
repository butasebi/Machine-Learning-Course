#!/usr/bin/env python
# coding: utf-8

# In[52]:


from torch.utils.data import DataLoader, Dataset
import PIL
import torchvision
import numpy
import pandas
from efficientnet_pytorch import EfficientNet
import torch 
import gc
import cv2

MAX_LOSS = 1e6


# In[53]:


class TrainDataset(Dataset):
    def __init__(self, labels_file, data_dir):
        self.filenames, self.labels = [], []
        self.data_dir = data_dir
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
        ])
        file = open(labels_file,'r')
        for line in file:
            filename, label = line.split(',')
            label = label[0]
            self.filenames.append(filename)
            self.labels.append(label)
        file.close()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = cv2.imread(self.data_dir + '/' + self.filenames[idx])
        image = self.transform(image)
        label = self.labels[idx]
        label = torch.from_numpy(numpy.array(int(label)))
        return (image, label)


# In[54]:


class TestDataset(Dataset):
    def __init__(self, labels_file, data_dir):
        self.filenames, self.labels = [], []
        self.data_dir = data_dir
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
        ])
        file = open(labels_file,'r')
        for line in file:
            filename = line.split()[0]
            self.filenames.append(filename)
        file.close()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = cv2.imread(self.data_dir + '/' + self.filenames[idx])
        image = self.transform(image)
        return image


# In[55]:


train_data = TrainDataset('C:/Users/gabi/Desktop/random/train.txt', 'C:/Users/gabi/Desktop/random/train+validation')
train_dataloader = DataLoader(dataset = train_data, batch_size = 32, shuffle = True)


# In[56]:



validation_data = TrainDataset('C:/Users/gabi/Desktop/random/validation.txt', 'C:/Users/gabi/Desktop/random/train+validation')
validation_dataloader = DataLoader(dataset=validation_data, batch_size = 32, shuffle = True)


# In[57]:


test_data = TestDataset('C:/Users/gabi/Desktop/random/test.txt', 'C:/Users/gabi/Desktop/random/test')
test_dataloader = DataLoader(dataset=test_data, batch_size = 32)


# In[58]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
eff_model = EfficientNet.from_pretrained('efficientnet-b0', num_classes = 7)
eff_model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(eff_model.parameters(), lr = 1e-4)

best_loss = MAX_LOSS

gc.collect()
torch.cuda.empty_cache()


# In[63]:


for epoch in range(100):
    if epoch + 1 == 3:
        optimizer = torch.optim.Adam(eff_model.parameters(), lr = 1e-5)
        
    if epoch + 1 == 5:
        optimizer = torch.optim.Adam(eff_model.parameters(), lr = 1e-6)
        
    train_loss = 0.0    
    eff_model.train()
    for useless_id, (images_batch, labels_batch) in enumerate(train_dataloader):
        if useless_id % 30 == 0:
            print(useless_id)
            
        optimizer.zero_grad()
        
        labels_batch = labels_batch.type(torch.LongTensor)
        
        images_batch = images_batch.to(device)
        labels_batch = labels_batch.to(device)
        
        labels_predictions = eff_model(images_batch)
        
        loss = criterion(labels_predictions, labels_batch)
        loss.backward()
        
        optimizer.step()
        
        train_loss = train_loss + loss.item()
        
    validation_loss = 0.0
    eff_model.eval()
    
    with torch.no_grad():
        for useless_id, (images_batch, labels_batch) in enumerate(validation_dataloader):
            labels_batch = labels_batch.type(torch.LongTensor)
            images_batch, labels_batch = images_batch.to(device), labels_batch.to(device)
            labels_predictions = eff_model(images_batch)
            loss = criterion(labels_predictions, labels_batch)
            validation_loss += loss.item()

    train_loss /= len(train_dataloader.dataset)
    validation_loss /= len(validation_dataloader.dataset)
    print(f'Epoch: {epoch} Train Loss: {train_loss} Validation Loss: {validation_loss}')
        
    if validation_loss < best_loss:
        checkpoint = {'checkpoint': eff_model.state_dict()}
        torch.save(checkpoint, './checkpoint.pt')
        print(f'Checkpoint reached!Validation loss modified from {best_loss} to {validation_loss}')
        best_loss = validation_loss


# In[64]:


from sklearn.metrics import accuracy_score, confusion_matrix
from torch import nn
import matplotlib.pyplot as plt
import torch
checkpoint = torch.load('./checkpoint.pt')
eff_model.load_state_dict(checkpoint['checkpoint'])
eff_model.eval()
smax = nn.Softmax(dim=1)

with torch.no_grad():
    labels_true = []
    labels_predictions_extended = []
    for batch_idx, (images_batch, labels_batch) in enumerate(validation_dataloader):
        images_batch, labels_batch = images_batch.to(device), labels_batch.to(device)
        labels_predictions = eff_model(images_batch).cpu()
        labels_predictions = smax(labels_predictions)
        labels_predictions = numpy.argmax(labels_predictions, axis=1)
        labels_predictions_extended.extend(labels_predictions)
        labels_true.extend(labels_batch.cpu())
        
accuracy = accuracy_score(labels_predictions_extended, labels_true)    
print(f'Validation accuracy: {accuracy}')


# In[15]:


import seaborn as sns
from sklearn.metrics import confusion_matrix
cf_matrix = confusion_matrix(labels_true, labels_predictions_extended)
sns.heatmap(cf_matrix, annot=True, fmt='g')
print(cf_matrix)


# In[65]:


with torch.no_grad():
    labels_predictions_extended = []
    for batch_idx, image_batch in enumerate(test_dataloader):
        image_batch = image_batch.to(device)
        label_predictions = eff_model(image_batch).cpu()
        label_predictions = smax(label_predictions)
        label_predictions = numpy.argmax(label_predictions, axis=1)
        labels_predictions_extended.extend(label_predictions)

file = open('C:/Users/gabi/Desktop/random/test.txt','r')
submission_file = open('./effnetb0_test.txt', 'w')
submission_file.write('id,label\n')

for i, line in enumerate(file):
    filename = line.split()[0]
    label = str(labels_predictions_extended[i].item())
    submission_file.write(filename + ',' + label + '\n')

file.close()
submission_file.close()


# In[ ]:




