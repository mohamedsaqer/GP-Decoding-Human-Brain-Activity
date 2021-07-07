import numpy as np
import torch.utils.data as data
import torch
import tensorflow as tf

class eegloader(data.Dataset):
	def __init__(self, data_path, labels, data_dir='./', dlen=160, stpt=320, nch=128):

		data = torch.load(data_dir + data_path)
		self.mean = data['means']
		self.stdev = data['stddevs']
		self.labels = labels
		self.data = []
    
		for l in self.labels:
			self.data.append(data['dataset'][l])

		assert len(self.data)==len(self.labels)
		self.dlen = dlen
		self.stpt = stpt
		self.nch = nch

	def __len__(self):
		return len(self.data)
	
	def __getitem__(self, idx):
		nch  = self.nch
		dlen = self.dlen 
		stpt = self.stpt
	
		x = np.zeros((nch,dlen))
		y = self.data[idx]['label']
		s = self.data[idx]['subject'] 
		
		x = torch.from_numpy(x)
		x[:,:min(int(self.data[idx]['eeg'].shape[1]),dlen)] = self.data[idx]['eeg'][:,stpt:stpt+dlen]
		x = x.type(torch.FloatTensor).sub(self.mean.expand(nch,dlen))/ self.stdev.expand(nch,dlen)

		return x,y,s

def load_data(data_path, labels, path ,dlen , stpt):
  


    x = eegloader(data_path, labels, path ,dlen, stpt)
    eeg = []
    y_labels = []
    x_train=[]
    y_train=[]
    x_test=[]
    y_test=[]
    x_val =[]
    y_val =[]
    train_size = int(len(x)*0.8)
    val_size = int(len(x)*0.1)
    test_size = len(x) - train_size - val_size

    for i in range(len(x)):
        eeg.append(tf.make_ndarray(tf.make_tensor_proto(x[i][0])))
        y_labels.append(x[i][1])

    eeg = np.transpose(np.array(eeg),(0,2,1))
    y_labels = np.array(y_labels)


    x_train = np.array(eeg[0:train_size])
    y_train = np.array(y_labels[0:train_size])
    x_test = np.array(eeg[train_size:train_size+test_size])
    y_test = np.array(y_labels[train_size:train_size+test_size])
    x_val = np.array(eeg[train_size+test_size:len(x)])
    y_val = np.array(y_labels[train_size+test_size:len(x)]) 
                      
    del(x,eeg,y_labels)
    return (x_train,y_train,x_test,y_test,x_val,y_val)

path = '/content/drive/My Drive/Dataset/eeg/'
data_path = 'eeg_signals_128_sequential_band_all_with_mean_std.pth'
diff = 200       ## diff
strt = 50   ## star of time point


data1 = torch.load(path + data_path)
dst = data1['dataset']
labels = []

for i in range(len(dst)):
  labels.append(dst[i]['image'])

labels = np.argsort(labels) 
del(data1,dst)
x_train,y_train,x_test,y_test,x_val,y_val = load_data(data_path, labels, path,dlen=diff, stpt=strt)

def data_load():
    return(load_data(data_path, labels, path,dlen=diff, stpt=strt))
