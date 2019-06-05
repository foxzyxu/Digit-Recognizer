import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop

train = pd.read_csv('train.csv')
#print(train.isnull().any().describe())
test = pd.read_csv('test.csv')
#print(test.isnull().any().describe())
Y_train = train['label']
#s1 = sns.countplot(Y_train)#看看标签的分布情况，较为平均
train = train.drop(labels = ['label'],axis = 1)
#简单的标准化
train = train / 255.0
test = test / 255.0
X_train = train.values.astype('float32').reshape(-1,28,28,1)
X_test = test.values.astype('float32').reshape(-1,28,28,1)
Y_train = Y_train = to_categorical(Y_train, num_classes = 10)
del train
del test
X_train, X_train_test, Y_train, Y_train_test = train_test_split(X_train, Y_train, test_size = 0.1, random_state=55, stratify = Y_train)
#p1 = plt.imshow(X_train[0][:,:,0])#看看photo

#开始CNN
model = Sequential()
#卷积和池化
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
#卷积和池化
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))#正则项
#fully connected neural network
model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation = 'softmax'))
#学习策略
model.compile(optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0),
              loss = 'categorical_crossentropy', metrics=['accuracy'])
#第一次用CNN就简单点用.fit,时间不多，随便训练个3轮试一试。
history = model.fit(X_train, Y_train, batch_size = 86, epochs = 3,
                  validation_data = (X_train_test, Y_train_test), verbose = 2)

#模型的调参与收敛性，省略
#预测和输出结果
results = model.predict(X_test)
results = np.argmax(results,axis = 1)
results = pd.Series(results,name='Label')
submission = pd.concat([pd.Series(range(1,28001),name = 'ImageId'),results],axis = 1)
submission.to_csv('Digit Recognizer.csv',index=False)