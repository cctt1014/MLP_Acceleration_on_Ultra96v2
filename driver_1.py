from pynq import Overlay
from pynq import allocate
import pynq.lib.dma
import numpy as np
import time

class Model:
    def __init__(self, bitfile, paramfile):
        self.overlay = Overlay(bitfile)
        self.dma = self.overlay.axi_dma_1
        
        f = open(paramfile, "r")
        self.params = f.read().split(',')
        for i in range(len(self.params)):
            self.params[i] = float(self.params[i])
        self.numofparams = len(self.params)
        
        self.input_buffer = allocate(shape=(self.numofparams+12,), dtype=np.float32)
        self.res = allocate(shape=(1,), dtype=np.float32)
        
        
    def classify(self, input_x):
        for i in range(12):
            self.input_buffer[self.numofparams+i] = input_x[i]
        self.dma.sendchannel.transfer(self.input_buffer)
        self.dma.recvchannel.transfer(self.res)
        self.dma.sendchannel.wait()
        self.dma.recvchannel.wait()
        return int(self.res[0])

def main():
    # Sample of using the Model class above
    
    # Initialize the model
    mlp = Model("design_1_1.bit", "out.csv")
    # Fake input here, shld be a list with 12 floats
    input_x = [12.0] * 12
    # classify function will return a class index (integer) with highest probability
    mlp.classify(input_x)
    
    
    # speed testing
    time_sum = 0
    for i in range(1000):
        start_time = time.time()
        mlp.classify(input_x)
        time_sum += (time.time()-start_time)*1000  # in ms

    average = time_sum / 1000.0
    print("Average time spent on HW data processing: ", (average), " ms")
    
if __name__ == "__main__":
    main()
