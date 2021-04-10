from pynq import Overlay
from pynq import allocate
import pynq.lib.dma
import numpy as np
from sklearn.preprocessing import StandardScaler
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt 
from os import listdir
from os.path import isfile, join
import scipy.stats as stats

INPUT_SIZE = 120
OUTPUT_SIZE = 3

data_frequency = 10 #50hz
frame_size = data_frequency * 2
sliding = data_frequency * 1

motion_data_directory = "Logs for week 11"
file_path = './' + motion_data_directory
all_files = [f for f in listdir(file_path) if isfile(join(file_path, f))]

def obtain_side_and_action_df(side, dance):
  df = pd.read_csv('./trial2/' + dance + '/' + side + '.csv')
  temp = []
  for i in df.columns:
    temp.append(side + '_' + i)
  df.columns = temp
  return df

def join_df(left, right):
  return pd.concat([left, right], axis=1)

def get_joined_df(move):
  temp_df_list = []
  for file in all_files:
    if move in file:
      temp_df = pd.read_csv(file_path + '/' + file)
      temp_df_list.append(temp_df)
  
  df = pd.concat(temp_df_list)
  return df

def get_frames(df, frame_size, sliding):
  num_features = 6

  frames = []
  labels = []
  for i in range(0, len(df) - frame_size, sliding):
    accel_x = df['accel.x'].values[i:i+frame_size]
    accel_y = df['accel.y'].values[i:i+frame_size]
    accel_z = df['accel.z'].values[i:i+frame_size]
    gyro_x = df['gyro.x'].values[i:i+frame_size]
    gyro_y = df['gyro.y'].values[i:i+frame_size]
    gyro_z = df['gyro.z'].values[i:i+frame_size]

    label = stats.mode(df['action_label'][i:i+frame_size])[0][0]
    frames.append([accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z])
    labels.append(label)
  
  frames = np.asarray(frames).reshape(-1, frame_size, num_features)
  labels = np.asarray(labels)
  return frames, labels

class Model:
    def __init__(self, bitfile, paramfile):
        # Initialize Overlay & DMA
        self.overlay = Overlay(bitfile)
        self.dma = self.overlay.axi_dma_0
        
        # Load weights and bias
        f = open(paramfile, "r")
        self.params = f.read().split(',')
        for i in range(len(self.params)):
            self.params[i] = float(self.params[i])
        self.numofparams = len(self.params)
        
        # Setup feature extraction
        self.scaler = StandardScaler()
        self.extracted_data = []
        
        # Initialize DMA buffer
        self.input_buffer = allocate(shape=(self.numofparams+INPUT_SIZE,), dtype=np.float32)
        for i in range(self.numofparams):
            self.input_buffer[i] = self.params[i]
        self.res = allocate(shape=(2*OUTPUT_SIZE,), dtype=np.float32)
    
    # raw_data is supposed to be a 20*6 numpy ndarray
    def preprocess(self, raw_data):
        self.extracted_data.clear()
        raw_data = self.scaler.fit_transform(raw_data)
        raw_data = raw_data.flatten()
        for i in range(len(raw_data)):    
            self.extracted_data.append(raw_data[i])
        
    def classify(self):
        for i in range(INPUT_SIZE):
            self.input_buffer[self.numofparams+i] = np.float32(self.extracted_data[i])
        self.dma.sendchannel.transfer(self.input_buffer)
        self.dma.recvchannel.transfer(self.res)
        self.dma.sendchannel.wait()
        self.dma.recvchannel.wait()
        
        return np.argmax(self.res)
        
def main():

    
    side_pump = get_joined_df('Sidepump')
    hair = get_joined_df('Hair')
    gun = get_joined_df('Gun')
    
    movement_list = [side_pump, hair, gun]
    # movement_list = [downstairs_df_list]
    action_reference = [1,2,3]
    movement = ["side_pump", "hair", "gun"]
    
    df_combined = pd.DataFrame(columns=[ 
       'accel.x', 'accel.y',
       'accel.z', 'gyro.x', 'gyro.y', 'gyro.z',
       'action'])

    for i in range(len(movement_list)):
      action = movement_list[i]
      action['action'] = movement[i]
      frames = [df_combined, action]
      df_combined = pd.concat(frames)

    df_temp = pd.DataFrame()
    df_temp["action"] = df_combined['action']
    df_combined = df_combined.drop(['action'], axis=1)
    df_combined = df_combined.drop(['timestamp'], axis=1)
    df_combined['action'] = df_temp['action']
    
    df_combined = df_combined.reset_index()
    df_combined = df_combined.drop(columns=['index'])
    df_combined
    from sklearn.preprocessing import LabelEncoder

    label = LabelEncoder()
    df_combined['action_label'] = label.fit_transform(df_combined['action'])
    
    col_names = []
    for col_name in df_combined.columns:
      col_names.append(col_name)

    X = df_combined[col_names[0:6]]
    y = df_combined['action_label']

    from sklearn.preprocessing import  StandardScaler

    scaler = StandardScaler()

    # X = scaler.fit_transform(X)
    X = pd.DataFrame(data = X, columns = col_names[0:6])
    X['action_label'] = y.values
    
    X, y = get_frames(X, frame_size, sliding)

    mlp = Model("bitstreams/mlpv3_1.bit", "mlpv3.csv")
    correct = 0
    start_time = time.time()
    for i in range(X.shape[0]):
        input_x = X[i]
        mlp.preprocess(input_x)
        pred = mlp.classify()
    #     print(pred,y[i])
        if (pred == y[i]):
            correct += 1
    print("-----%s ms in average elapsed for each transfer-----" %(1000*(time.time()-start_time)/X.shape[0]))
    print("TOTAL ACCURACY: ", correct/float(X.shape[0]))
    
if __name__ == '__main__':
    main()
    