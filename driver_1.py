from pynq import Overlay
from pynq import allocate
import pynq.lib.dma
from pynq import Xlnk
import numpy as np
import csv
import time


overlay = Overlay("design_1_1.bit")

WEIGHTS_01 = 12
WEIGHTS_12 = 4
WEIGHTS_23 = 2
WEIGHTS_34 = 2
WEIGHTS_45 = 2
WEIGHTS_56 = 4
WEIGHTS_67 = 12

dma = overlay.axi_dma_1

f = open("out.csv", "r")

weights = f.read().split(',')

for i in range(len(weights)):
    weights[i] = float(weights[i])

### Dummy input data here ###
input_x = [1.0] * WEIGHTS_67

input_list = weights + input_x

input_buffer = allocate(shape=(158,), dtype=np.float32)
# input_list = np.array([0.0]*158)
for i in range(158):
    input_buffer[i] = input_list[i]
    
res = allocate(shape=(1,), dtype=np.float32)

time_sum = 0
for i in range(1000):
    start_time = time.time()

    dma.sendchannel.transfer(input_buffer)
    dma.recvchannel.transfer(res)
    dma.sendchannel.wait()
    dma.recvchannel.wait()

#     print("-----%s ms elapsed for transfer-----" %(1000*(time.time()-start_time)))
    time_sum += (time.time()-start_time)*1000 #in ms

average = time_sum / 1000.0
print("Average time spent on HW data processing: ", (average), " ms")
print("Result: ", res)
