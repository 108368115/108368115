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
##### -->執行檔
##### [Data source](https://www.kaggle.com/c/machine-learning-ntut-2019-speech-recognition/data)
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
##### 4. 載入訓練數據集
```python
train_df, valid_df = load_data("H:/MachineLearning/speech-recognition/ntut speech recognition/train/train/", test_size)
```
##### 5. 讀取音檔並生成訓練集和驗證集。
```python
def load_data(path, test_size):
    
    POSSIBLE_LABELS = glob(path + 'audio/*/*wav')                      讀取./audio路徑下所有的.wav音檔
    LABEL = []
    columns_list = ['label', 'label_id', 'user_id', 'wav_file']        設置列表項目 
    
    for file_path in POSSIBLE_LABELS:
        pattern = re.compile("(.+\/)?(\w+)\/([^_]+)_.+wav")
        r = re.match(pattern, file_path)
        
        r_label, r_file = r.group(2), r.group(3)
        if (r_label == '_background_noise_'):
            r_label = 'silence'
        if r_label not in labels:
            r_label = 'unknown'
            
        label_id = name2id[r_label]
        
        sample = [r_label, label_id, r_file, file_path]
        LABEL.append(sample)
            
    
    np_sample = np.array(LABEL)
    
    train, valid = train_test_split(np_sample, test_size = test_size)  從./train中按比例隨機選取訓練數據集和驗證數據集
    
    train_df = pd.DataFrame(train, columns = columns_list)
    valid_df = pd.DataFrame(valid, columns = columns_list)
    
    print('There are %d train and %d val samples' %(train.shape[0], valid.shape[0]))  印出訓練數據集總數和驗證數據集總數
    return train_df, valid_df
```
```There are 43790 train and 10948 val samples```
##### 6. 訓練集標籤。
```python
train_df.label.value_counts()
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
##### 8. 處理.wav檔。
```python
def read_wav_file(fname):
    _, wav = wavfile.read(fname)
    wav = wav.astype(np.float32) / np.iinfo(np.int16).max
    return wav
```
##### 9. 。
```python
silence_data = np.concatenate([read_wav_file(x) for x in silence_files.wav_file.values])
silence_data.shape
```
##### 10. 。
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
##### 11. 。
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
##### 12. 。
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
##### 13. 建立模型。
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
##### 14. 。
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
##### 16. 。
```python
model.load_weights('./train/weight/starter.hdf5')
```
##### 17. 。
```python
test_paths = glob(os.path.join('./', 'test/*wav'))
```
##### 18. 。
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
```
##### 19. 。
```python
predictions = model.predict_generator(test_generator(100), int(np.ceil(len(test_paths)/64)))

classes = np.argmax(predictions, axis=1)

submission = dict()
for i in range(len(test_paths)):
    fname, label = os.path.basename(test_paths[i]), id2name[classes[i]]
    submission[fname] = label
```
##### 20. 。
```python
with open('starter_submission.csv', 'w') as fout:
    fout.write('fname,label\n')
    for fname, label in submission.items():
        fout.write('{},{}\n'.format(fname, label))
```
## 4.結果分析
