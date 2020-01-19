# Speech Recognition
![image](https://github.com/MachineLearningNTUT/application-108368115/blob/master/Speech.jpg)<br/><br/>
### 學號:108368115
### 姓名:莊旻儒
<br><br>
<br><br>
## 1.程式方塊圖
## 2.檔案說明
##### ./train -->訓練數據集
##### ./test  -->測試數據集
##### [Data source](https://drive.google.com/open?id=1aqGd61XBvthmiv3DRQbcMPVW3mwT7XGF)
## 3.用法說明
##### 1. 設定環境變數。
```python
#!/usr/bin/env python     調用/usr/bin/env下的python編譯器
# coding: utf-8           宣告編碼方式
```
##### 2. 引入函式庫。
```python
import os
import re
from glob import glob
import numpy as np
from IPython import get_ipython
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from scipy.io import wavfile
from scipy.signal import stft
import random
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Activation, BatchNormalization, GlobalAveragePooling2D, GlobalMaxPool2D, concatenate, Dense, Dropout
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from keras_tqdm import TQDMNotebookCallback
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import h5py
import wave
```
##### 3. 參數設置。
```python
labels = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence', 'unknown'] 設置12個分類標籤
id2name = {i: name for i, name in enumerate(labels)}
name2id = {name: i for i, name in id2name.items()}
test_size = 0.2     提取20%的訓練數據集作為驗證使用
```
##### 4. 讀取音檔並生成訓練集和驗證集。
```python
def load_data(path, test_size):                                        path變數=./train訓練數據集位址 
    
    POSSIBLE_LABELS = glob(path + 'audio/*/*wav')                      讀取./audio路徑下所有的.wav音檔
    LABEL = []
    columns_list = ['label', 'label_id', 'user_id', 'wav_file']        自設列表的列向名稱 
    
    for file_path in POSSIBLE_LABELS:                                  迴圈會依序從POSSIBLE_LABELS取得元素，並將元素指定給file_path，<br/>
                                                                       再執行迴圈裡的內容，直到序列每一元素都被取出過為止。
        pattern = re.compile("(.+(?:\/|\\\\))?(\w+)(?:\/|\\\\)([^_]+)_.+wav") 
        pattern(編譯時用的表達式字符串):正規表示式樣式，為用來描述或者匹配一系列符合某個句法規則的字串
        re.compile用來將正則表達式轉換爲模式對象
        使用( )括住的稱為子運算式，可以用來定義運算子的範圍和優先度，每個子運算式匹配的結果稱為群組(group)
        "\"將下一個字元標記為一個特殊字元
        "?"示前面的字元最多只可以出現一次
        "\w+"示匹配包括底線的任何單詞字元和加號本身字符1次，"+"等價於{1,}
        [^_]表示優先匹配除了"_"以外的字符串
        "+"表示前面的字元必須至少出現一次
        r = re.match(pattern, file_path)                                如果能夠在POSSIBLE_LABELS找到任意個匹配pattern的正規樣式，就返回一個相應的匹配對象
        r_label, r_file = r.group(2), r.group(3)                        通過group(表達式中捕獲群的數量)我們可以選擇匹配到的字符串中「需要」的部分                                       
        group(0)表示匹配到的整個字符串                                              
        group(數字)表示在()中的子字符串                                                  
        if (r_label == '_background_noise_'):
            r_label = 'silence'
        if r_label not in labels:
            r_label = 'unknown'
            
        label_id = name2id[r_label]
        
        sample = [r_label, label_id, r_file, file_path]
        LABEL.append(sample)
            
    
    np_sample = np.array(LABEL)
    
    train, valid = train_test_split(np_sample, test_size = test_size)    從./train中按比例隨機選取訓練數據集和驗證數據集
    
    train_df = pd.DataFrame(train, columns = columns_list)
    valid_df = pd.DataFrame(valid, columns = columns_list)
    
    print('There are %d train and %d val samples' %(train.shape[0], valid.shape[0]))  印出訓練數據集總數和驗證數據集總數
    return train_df, valid_df
```
```There are 43790 train and 10948 val samples```
##### 5. 載入訓練數據集。
```python
train_df, valid_df = load_data("H:/MachineLearning/speech-recognition/ntut speech recognition/train/train/", test_size)
```
##### 6. 訓練集標籤。
```python
train_df.label.value_counts()
```
```
unknown    27733
up          1655
yes         1616
down        1613
off         1611
no          1607
on          1601
left        1596
go          1591
stop        1586
right       1575
silence        6
Name: label, dtype: int64
```
##### 7. 訓練集對silence標籤進行分類。
```python
silence_files = train_df[train_df.label == 'silence']   符合silence標籤的指定給 silence_files 存放
train_df = train_df[train_df.label != 'silence']        不符合silence標籤的指定給 train_df 存放
print ("--------------------------------------")
print (silence_files)                                   印出符合silence標籤的音檔資訊
print ("--------------------------------------")
print (train_df)                                        印出不符合silence標籤的音檔資訊
```
```
--------------------------------------
silence_files:          label label_id   user_id  \
1232   silence       10      pink   
1935   silence       10     doing   
11657  silence       10     white   
20930  silence       10   running   
24215  silence       10      dude   
36127  silence       10  exercise   

                                                wav_file  
1232   E:/北科研究所/機器學習/Speech/speech recognition/ntut s...  
1935   E:/北科研究所/機器學習/Speech/speech recognition/ntut s...  
11657  E:/北科研究所/機器學習/Speech/speech recognition/ntut s...  
20930  E:/北科研究所/機器學習/Speech/speech recognition/ntut s...  
24215  E:/北科研究所/機器學習/Speech/speech recognition/ntut s...  
36127  E:/北科研究所/機器學習/Speech/speech recognition/ntut s...  
--------------------------------------
train_df:
          label label_id   user_id  \
0      unknown       11  7d8babdb   
1         left        4  6c9223bd   
2      unknown       11  45adf84a   
3         stop        8  f19c1390   
4      unknown       11  520e8c0e   
...        ...      ...       ...   
43785       go        9  590750e8   
43786  unknown       11  0e4d22f1   
43787      yes        0  1b63157b   
43788       up        2  73124b26   
43789  unknown       11  b16f2d0d   

                                                wav_file  
0      E:/北科研究所/機器學習/Speech/speech recognition/ntut s...  
1      E:/北科研究所/機器學習/Speech/speech recognition/ntut s...  
2      E:/北科研究所/機器學習/Speech/speech recognition/ntut s...  
3      E:/北科研究所/機器學習/Speech/speech recognition/ntut s...  
4      E:/北科研究所/機器學習/Speech/speech recognition/ntut s...  
...                                                  ...  
43785  E:/北科研究所/機器學習/Speech/speech recognition/ntut s...  
43786  E:/北科研究所/機器學習/Speech/speech recognition/ntut s...  
43787  E:/北科研究所/機器學習/Speech/speech recognition/ntut s...  
43788  E:/北科研究所/機器學習/Speech/speech recognition/ntut s...  
43789  E:/北科研究所/機器學習/Speech/speech recognition/ntut s...  

[43784 rows x 4 columns]
```
##### 8. 處理.wav檔。
```python
def read_wav_file(fname):
    _, wav = wavfile.read(fname)
    wav = wav.astype(np.float32) / np.iinfo(np.int16).max
    return wav
    silence_data = np.concatenate([read_wav_file(x) for x in silence_files.wav_file.values])
silence_data.shape
```
```(6390371,)```
##### 9. 定義模型架構。
```python
def process_wav_file(fname):
    wav = read_wav_file(fname)
    
    L = 16000  # 1 sec
    
    if len(wav) > L:
        i = np.random.randint(0, len(wav) - L)
        wav = wav[i:(i+L)]
    elif len(wav) < L:
        rem_len = L - len(wav)
        i = np.random.randint(0, len(silence_data) - rem_len)
        silence_part = silence_data[i:(i+L)]
        j = np.random.randint(0, rem_len)
        silence_part_left  = silence_part[0:j]
        silence_part_right = silence_part[j:rem_len]
        wav = np.concatenate([silence_part_left, wav, silence_part_right])
    
    specgram = stft(wav, 16000, nperseg = 400, noverlap = 240, nfft = 512, padded = False, boundary = None)
    phase = np.angle(specgram[2]) / np.pi
    amp = np.log1p(np.abs(specgram[2]))
    
    return np.stack([phase, amp], axis = 2)
```
##### 10. 定義訓練模型架構。
```python
def train_generator(train_batch_size):
    while True:
        this_train = train_df.groupby('label_id').apply(lambda x: x.sample(n = 1000))
        shuffled_ids = random.sample(range(this_train.shape[0]), this_train.shape[0])
        for start in range(0, len(shuffled_ids), train_batch_size):
            x_batch = []
            y_batch = []
            end = min(start + train_batch_size, len(shuffled_ids))
            i_train_batch = shuffled_ids[start:end]
            for i in i_train_batch:
                x_batch.append(process_wav_file(this_train.wav_file.values[i]))
                y_batch.append(this_train.label_id.values[i])
            x_batch = np.array(x_batch)
            y_batch = to_categorical(y_batch, num_classes = len(labels))
            yield x_batch, y_batch
```
##### 11. 定義驗證模型架構。
```python
def valid_generator(val_batch_size):
    while True:
        ids = list(range(valid_df.shape[0]))
        for start in range(0, len(ids), val_batch_size):
            x_batch = []
            y_batch = []
            end = min(start + val_batch_size, len(ids))
            i_val_batch = ids[start:end]
            for i in i_val_batch:
                x_batch.append(process_wav_file(valid_df.wav_file.values[i]))
                y_batch.append(valid_df.label_id.values[i])
            x_batch = np.array(x_batch)
            y_batch = to_categorical(y_batch, num_classes = len(labels))
            yield x_batch, y_batch
```
##### 12. 建立模型。
```python
x_in = Input(shape = (257,98,2))
x = BatchNormalization()(x_in)
for i in range(4):
    x = Conv2D(16*(2 ** i), (3,3))(x)
    x = Activation('elu')(x)
    x = BatchNormalization()(x)                                 
    x = MaxPooling2D((2,2))(x)
x = Conv2D(128, (1,1))(x)
x_branch_1 = GlobalAveragePooling2D()(x)                                                     
x_branch_2 = GlobalMaxPool2D()(x)
x = concatenate([x_branch_1, x_branch_2])
x = Dense(256, activation = 'relu')(x)
x = Dropout(0.5)(x)
x = Dense(len(labels), activation = 'softmax')(x)
#x = Dense(12, activation='sigmoid')(x)

model = Model(inputs = x_in, outputs = x)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
```
```
Model: "model_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 257, 98, 2)   0                                            
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 257, 98, 2)   8           input_1[0][0]                    
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 255, 96, 16)  304         batch_normalization_1[0][0]      
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 255, 96, 16)  0           conv2d_1[0][0]                   
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 255, 96, 16)  64          activation_1[0][0]               
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 127, 48, 16)  0           batch_normalization_2[0][0]      
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 125, 46, 32)  4640        max_pooling2d_1[0][0]            
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 125, 46, 32)  0           conv2d_2[0][0]                   
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 125, 46, 32)  128         activation_2[0][0]               
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, 62, 23, 32)   0           batch_normalization_3[0][0]      
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 60, 21, 64)   18496       max_pooling2d_2[0][0]            
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 60, 21, 64)   0           conv2d_3[0][0]                   
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 60, 21, 64)   256         activation_3[0][0]               
__________________________________________________________________________________________________
max_pooling2d_3 (MaxPooling2D)  (None, 30, 10, 64)   0           batch_normalization_4[0][0]      
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 28, 8, 128)   73856       max_pooling2d_3[0][0]            
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 28, 8, 128)   0           conv2d_4[0][0]                   
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, 28, 8, 128)   512         activation_4[0][0]               
__________________________________________________________________________________________________
max_pooling2d_4 (MaxPooling2D)  (None, 14, 4, 128)   0           batch_normalization_5[0][0]      
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 14, 4, 128)   16512       max_pooling2d_4[0][0]            
__________________________________________________________________________________________________
global_average_pooling2d_1 (Glo (None, 128)          0           conv2d_5[0][0]                   
__________________________________________________________________________________________________
global_max_pooling2d_1 (GlobalM (None, 128)          0           conv2d_5[0][0]                   
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 256)          0           global_average_pooling2d_1[0][0] 
                                                                 global_max_pooling2d_1[0][0]     
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 256)          65792       concatenate_1[0][0]              
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 256)          0           dense_1[0][0]                    
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 12)           3084        dropout_1[0][0]                  
==================================================================================================
Total params: 183,652
Trainable params: 183,168
Non-trainable params: 484
```
##### 13. 設定callbacks特定操作。
```python
callbacks = [EarlyStopping(monitor='val_loss',
                           patience=5,
                           verbose=1,
                           min_delta=0.01,
                           mode='min'),
             ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=3,
                               verbose=1,
                               epsilon=0.01,
                               mode='min'),
             ModelCheckpoint(monitor='val_loss',
                             filepath='./train/weight/starter.hdf5',
                             save_best_only=True,
                             save_weights_only=True,
                             mode='min'),
             TQDMNotebookCallback()]
```             
##### 15. 訓練模型。
```python
history = model.fit_generator(generator=train_generator(64),
                              steps_per_epoch=344,
                              epochs=20,
                              verbose=1,
                              callbacks=callbacks,
                              validation_data=valid_generator(64),
                              validation_steps=int(np.ceil(valid_df.shape[0]/64)))
```
```
Epoch 1/20
344/344 [==============================] - 76s - loss: 2.3986 - acc: 0.1352 - val_loss: 2.4137 - val_acc: 0.0535
Epoch 2/20
344/344 [==============================] - 69s - loss: 1.3059 - acc: 0.5475 - val_loss: 2.5465 - val_acc: 0.1973
Epoch 3/20
344/344 [==============================] - 68s - loss: 0.7153 - acc: 0.7631 - val_loss: 1.1454 - val_acc: 0.5693
...
```
##### 16. 儲存 Model Weight。
```python
model.load_weights('./train/weight/starter.hdf5')
```
##### 17. 載入測試數據集。
```python
test_paths = glob(os.path.join('./', 'test/*wav'))
```
##### 18. 預測 Label。
```python
def test_generator(test_batch_size):
    while True:
        for start in range(0, len(test_paths), test_batch_size):
            x_batch = []
            end = min(start + test_batch_size, len(test_paths))
            this_paths = test_paths[start:end]
            for x in this_paths:
                x_batch.append(process_wav_file(x))
            x_batch = np.array(x_batch)
            yield x_batch

predictions = model.predict_generator(test_generator(100), int(np.ceil(len(test_paths)/64)))

classes = np.argmax(predictions, axis=1)

submission = dict()
for i in range(len(test_paths)):
    fname, label = os.path.basename(test_paths[i]), id2name[classes[i]]
    submission[fname] = label
```
##### 20. 將預測結果儲存成 csv檔。
```python
with open('starter_submission.csv', 'w') as fout:
    fout.write('fname,label\n')
    for fname, label in submission.items():
        fout.write('{},{}\n'.format(fname, label))
```
