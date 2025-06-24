# Matrix Algebra
import numpy as np
from   numba import vectorize, jit, prange
from scipy.signal import butter, sosfilt, lfilter

import cv2
import math
import time
import logging
from threading import Lock

from PyQt5.QtCore import QObject, QTimer, QThread, pyqtSignal, pyqtSlot, QSignalMapper
from PyQt5.QtWidgets import QLineEdit, QSlider, QCheckBox, QLabel

class QProcessWorker(QObject):
    """ 
    Process Worker Class

    Signals      
        = For processWorker
        NEED TO DEVELOP
    Slots
      on_changeBinning
      on_processRequest
      
    """
    
    #We should do the processing here and not in the datacube class
    ######################################
    
    @pyqtSlot(list)
    def on_changeBinning(self, binning):
       self

class QDataCube(QObject):
    # emit just the raw cube when it fills up
    dataCubeReady = pyqtSignal(np.ndarray)

    def __init__(self, 
                 depth=14, 
                 height=540, 
                 width=720, 
                 parent=None
        ):

        super().__init__(parent)
        self.depth = depth
        self.data = np.zeros((depth, height, width), dtype=np.uint8)
        self._idx = 0
        self._lock = Lock()

    @pyqtSlot(np.ndarray)
    def add(self, image):
        """Called in the capture thread.  Only stores & emits."""
        with self._lock:
            self.data[self._idx] = image
            self._idx += 1
            if self._idx >= self.depth:
                # copy once so the worker can immediately reuse self.data
                cube_copy = self.data.copy()
                self._idx = 0
        if 'cube_copy' in locals():
            self.dataCubeReady.emit(cube_copy)

    @pyqtSlot()
    def clear(self):
        """Clear the data cube."""
        with self._lock:
            self.data.fill(0)
            self._idx = 0

    @property
    def capacity(self):
        """Return the capacity of the data cube."""
        return self.data.shape
    
    @property
    def index(self):
        """Return the current size of the data cube."""
        return self._idx

class QDataCubeProcess(QObject):
    """ 
    Data Cube Class
      sort()      sort so that lowest intensity is first image in stack
      bgflat()    subtract background, multiply flatfield
      bin2        binning 2x2 (explicit code is faster than general binning with slicing and summing in numpy)
      bin3        binning 3x3
      bin4        binning 4x4
      bin5        binning 5x5
      bin6        binning 6x6
      bin9        binning 9x9
      bin10       binning 10x10
      bin12       binning 12x12
      bin15       binning 15x15
      bin18       binning 18x18
      bin20       binning 20x20
    """

    processedCubeReady = pyqtSignal(np.ndarray)                                          # we have a complete datacube
    
    def __init__(self, flatfield=None, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.ff = flatfield

    @pyqtSlot(np.ndarray)
    def process(self, cube: np.ndarray, binning: int = 1):
        """
        Full pipeline: sort, bgflat, bin (optional), then emit.
        """
        try:
            sorted_cube = self.sort(cube)
            if self.ff is not None:
                bg = sorted_cube[0].astype(np.uint8)
                ff = self.ff
                corrected = self.bgflat8(sorted_cube.astype(np.uint8), bg, ff)
            else:
                corrected = sorted_cube
            if binning > 1:
                if binning == 2:
                    corrected = self.bin2(corrected)
                elif binning == 3:
                    corrected = self.bin3(corrected)
                elif binning == 4:
                    corrected = self.bin4(corrected)
                elif binning == 5:
                    corrected = self.bin5(corrected)
                elif binning == 6:
                    corrected = self.bin6(corrected)
                elif binning == 8:
                    corrected = self.bin8(corrected)
                elif binning == 9:
                    corrected = self.bin9(corrected)
                elif binning == 10:
                    corrected = self.bin10(corrected)
                elif binning == 12:
                    corrected = self.bin12(corrected)
                elif binning == 15:
                    corrected = self.bin15(corrected)
                elif binning == 18:
                    corrected = self.bin18(corrected)
                elif binning == 20:
                    corrected = self.bin20(corrected)
            self.processedCubeReady.emit(corrected)
        except Exception as e:
            self.logger.error(f"Processing failed: {e}")

    def sort(self,data: np.array, delta: tuple = (64,64)) -> np.array:
        """ Sorts data cube so that first image is the one with lowest intensity (background) """
        # create intensity reading for each image in the stack
        dx, dy = delta                                      # take intensity values at delta x and delta y intervals
        depth, width, height = self.data.shape  
        inten = np.sum(self.data[:,::dx,::dy], axis=(1,2)) # intensities at selected points in image
        # create sorting index        
        background_indx = int(np.argmin(inten))             # lowest intensity
        indx = np.roll(np.arange(depth), -background_indx)  # roll array index so that background is first
        # data sorted
        self.data = self.data[indx,:,:]                     # rearrange data cube
    
    # # Faltfield Correction and Background removal
    # #            result stack  bg     ff
    # @vectorize(['uint16(uint8, uint8, uint16)'], nopython=True, fastmath=True, cache=True)
    # def bgflat8(data_cube, background, flatfield):
    #     """Background removal, flat field correction, white balance """
    #     return np.multiply(np.subtract(data_cube, background), flatfield) # 8bit subtraction, 16bit multiplication

    # Background-flat correction for 8-bit
    @staticmethod
    @jit("uint16[:,:,:](uint8[:,:,:], uint8[:,:], uint16[:,:,:])", nopython=True, fastmath=True, cache=True, parallel=True)
    def bgflat8(data_cube, background, flatfield):
        depth, h, w = data_cube.shape
        out = np.empty_like(data_cube, dtype=np.uint16)
        for i in prange(depth):
            for y in range(h):
                for x in range(w):
                    val = data_cube[i, y, x] - background[y, x]
                    if val < 0:
                        val = 0
                    out[i, y, x] = val * flatfield[i, y, x]
        return out
    
    # Faltfield Correction and Background removal
    # #            result stack  bg     ff
    # @vectorize(['uint32(uint16, uint16, uint16)'], nopython=True, fastmath=True, cache=True)
    # def bgflat16(data_cube, background, flatfield):
    #     """Background removal, flat field correction, white balance """
    #     return np.multiply(np.subtract(data_cube, background), flatfield) # 8bit subtraction, 16bit multiplication
    # Background-flat correction for 16-bit
    @staticmethod
    @jit("uint32[:,:,:](uint16[:,:,:], uint16[:,:], uint16[:,:,:])", nopython=True, fastmath=True, cache=True, parallel=True)
    def bgflat16(data_cube, background, flatfield):
        depth, h, w = data_cube.shape
        out = np.empty_like(data_cube, dtype=np.uint32)
        for i in prange(depth):
            for y in range(h):
                for x in range(w):
                    val = int(data_cube[i, y, x]) - int(background[y, x])
                    if val < 0:
                        val = 0
                    out[i, y, x] = val * flatfield[i, y, x]
        return out
    
    # General purpose binning, this is 3 times slower compared to the routines below
    # @jit(nopython=True, fastmath=True, cache=True)
    # def rebin(arr, bin_x, bin_y, dtype=np.uint16):
    #     # https://stackoverflow.com/questions/36063658/how-to-bin-a-2d-array-in-numpy
    #     m,n,o = np.shape(arr)
    #     shape = (m//bin_x, bin_x, n//bin_y, bin_y, o)
    #     arr_ = arr.astype(dtype)
    #     return arr_.reshape(shape).sum(3).sum(1)

    # Binning 2 pixels of the 8bit images
    # @staticmethod
    # @jit(nopython=True, fastmath=True, parallel=True, cache=True)
    # def _bin2(arr):
    #     m, n, o = arr.shape
    #     tmp = np.empty((m//2, n, o), dtype=np.uint16)
    #     out = np.empty((m//2, n//2, o), dtype=np.uint16)
    #     for i in prange(m//2):
    #         tmp[i] = arr[2*i] + arr[2*i+1]
    #     for j in prange(n//2):
    #         out[:, j] = tmp[:, 2*j] + tmp[:, 2*j+1]
    #     return out
    
    @staticmethod
    @jit(nopython=True, fastmath=True, cache=True, parallel=True)
    def _bin2(arr):
        m, n, o = arr.shape
        out = np.empty((m // 2, n // 2, o), dtype=np.uint16)
        for i in prange(m // 2):
            for j in prange(n // 2):
                for k in range(o):
                    out[i, j, k] = (
                        arr[2*i, 2*j,   k] + arr[2*i+1, 2*j,   k] +
                        arr[2*i, 2*j+1, k] + arr[2*i+1, 2*j+1, k]
                    )
        return out
    
    # Binning 3 pixels of the 8bit images
    # @staticmethod
    # @jit(nopython=True, fastmath=True, parallel=True, cache=True)
    # def bin3(arr_in):
    #     m,n,o   = np.shape(arr_in)
    #     arr_tmp = np.empty((m//3,n,o), dtype='uint16')
    #     arr_out = np.empty((m//3,n//3,o), dtype='uint16')
    #     for i in prange(m//3):
    #         arr_tmp[i,:,:] =  arr_in[i*3,:,:] +  arr_in[i*3+1,:,:] +  arr_in[i*3+2,:,:] 
    #     for j in prange(n//3):
    #         arr_out[:,j,:] = arr_tmp[:,j*3,:] + arr_tmp[:,j*3+1,:] + arr_tmp[:,j*3+2,:] 
    #     return arr_out
    @staticmethod
    @jit(nopython=True, fastmath=True, parallel=True, cache=True)
    def bin3(arr):
        m, n, o = arr.shape
        tmp = np.empty((m//3, n, o), dtype=np.uint16)
        out = np.empty((m//3, n//3, o), dtype=np.uint16)
        for i in prange(m//3):
            tmp[i] = arr[3*i] + arr[3*i+1] + arr[3*i+2]
        for j in prange(n//3):
            out[:, j] = tmp[:, 3*j] + tmp[:, 3*j+1] + tmp[:, 3*j+2]
        return out
    
    # Binning 4 pixels of the 8bit images
    # @staticmethod
    # @jit(nopython=True, fastmath=True, parallel=True, cache=True)
    # def bin4(arr_in):
    #     m,n,o   = np.shape(arr_in)
    #     arr_tmp = np.empty((m//4,n,o), dtype='uint16')
    #     arr_out = np.empty((m//4,n//4,o), dtype='uint16')
    #     for i in prange(m//4):
    #         arr_tmp[i,:,:] =  arr_in[i*4,:,:] +  arr_in[i*4+1,:,:] +  arr_in[i*4+2,:,:] +  arr_in[i*4+3,:,:]
    #     for j in prange(n//4):
    #         arr_out[:,j,:] = arr_tmp[:,j*4,:] + arr_tmp[:,j*4+1,:] + arr_tmp[:,j*4+2,:] + arr_tmp[:,j*4+3,:]
    #     return arr_out
    @staticmethod
    @jit(nopython=True, fastmath=True, parallel=True, cache=True)
    def bin4(arr):
        m, n, o = arr.shape
        tmp = np.empty((m//4, n, o), dtype=np.uint16)
        out = np.empty((m//4, n//4, o), dtype=np.uint16)
        for i in prange(m//4):
            tmp[i] = arr[4*i] + arr[4*i+1] + arr[4*i+2] + arr[4*i+3]
        for j in prange(n//4):
            out[:, j] = tmp[:, 4*j] + tmp[:, 4*j+1] + tmp[:, 4*j+2] + tmp[:, 4*j+3]
        return out
    
    # Binning 5 pixels of the 8bit images
    # @staticmethod
    # @jit(nopython=True, fastmath=True, parallel=True, cache=True)
    # def bin5(arr_in):
    #     m,n,o   = np.shape(arr_in)
    #     arr_tmp = np.empty((m//5,n,o), dtype='uint16')
    #     arr_out = np.empty((m//5,n//5,o), dtype='uint16')
    #     for i in prange(m//5):
    #         arr_tmp[i,:,:] =  arr_in[i*5,:,:] +  arr_in[i*5+1,:,:] +  arr_in[i*5+2,:,:] +  arr_in[i*5+3,:,:] +  arr_in[i*5+4,:,:]
    #     for j in prange(n//5):
    #         arr_out[:,j,:] = arr_tmp[:,j*5,:] + arr_tmp[:,j*5+1,:] + arr_tmp[:,j*5+2,:] + arr_tmp[:,j*5+3,:] + arr_tmp[:,j*5+4,:] 
    #     return arr_out
    @staticmethod
    @jit(nopython=True, fastmath=True, parallel=True, cache=True)
    def bin5(arr):
        m, n, o = arr.shape
        tmp = np.empty((m//5, n, o), dtype=np.uint16)
        out = np.empty((m//5, n//5, o), dtype=np.uint16)
        for i in prange(m//5):
            tmp[i] = arr[5*i] + arr[5*i+1] + arr[5*i+2] + arr[5*i+3] + arr[5*i+4]
        for j in prange(n//5):
            out[:, j] = tmp[:, 5*j] + tmp[:, 5*j+1] + tmp[:, 5*j+2] + tmp[:, 5*j+3] + tmp[:, 5*j+4]
        return out
    
    # Binning 6 pixels of the 8bit images
    # @staticmethod
    # @jit(nopython=True, fastmath=True, parallel=True, cache=True)
    # def bin6(arr_in):
    #     m,n,o   = np.shape(arr_in)
    #     arr_tmp = np.empty((m//6,n,o), dtype='uint16')
    #     arr_out = np.empty((m//6,n//6,o), dtype='uint16')
    #     for i in prange(m//6):
    #         arr_tmp[i,:,:] =  arr_in[i*6,:,:] +  arr_in[i*6+1,:,:] +  arr_in[i*6+2,:,:] +  arr_in[i*6+3,:,:] +  arr_in[i*6+4,:,:]  \
    #                        +  arr_in[i*6+5,:,:]
    #     for j in prange(n//6):
    #         arr_out[:,j,:] = arr_tmp[:,j*6,:] + arr_tmp[:,j*6+1,:] + arr_tmp[:,j*6+2,:] + arr_tmp[:,j*6+3,:] + arr_tmp[:,j*6+4,:] \
    #                        + arr_tmp[:,j*6+5,:]  
    #     return arr_out
    @staticmethod
    @jit(nopython=True, fastmath=True, parallel=True, cache=True)
    def bin6(arr):
        m, n, o = arr.shape
        tmp = np.empty((m//6, n, o), dtype=np.uint16)
        out = np.empty((m//6, n//6, o), dtype=np.uint16)
        for i in prange(m//6):
            tmp[i] = arr[6*i] + arr[6*i+1] + arr[6*i+2] + arr[6*i+3] + arr[6*i+4] + arr[6*i+5]
        for j in prange(n//6):
            out[:, j] = tmp[:, 6*j] + tmp[:, 6*j+1] + tmp[:, 6*j+2] + tmp[:, 6*j+3] + tmp[:, 6*j+4] + tmp[:, 6*j+5]
        return out

    # Binning 8 pixels of the 8bit images
    @staticmethod
    @jit(nopython=True, fastmath=True, parallel=True, cache=True)
    def bin8(arr):
        m, n, o = arr.shape
        tmp = np.empty((m//8, n, o), dtype=np.uint16)
        out = np.empty((m//8, n//8, o), dtype=np.uint16)
        for i in prange(m//8):
            tmp[i] = (
                arr[8*i] + arr[8*i+1] + arr[8*i+2] + arr[8*i+3]
                + arr[8*i+4] + arr[8*i+5] + arr[8*i+6] + arr[8*i+7]
            )
        for j in prange(n//8):
            out[:, j] = (
                tmp[:, 8*j] + tmp[:, 8*j+1] + tmp[:, 8*j+2] + tmp[:, 8*j+3]
                + tmp[:, 8*j+4] + tmp[:, 8*j+5] + tmp[:, 8*j+6] + tmp[:, 8*j+7]
            )
        return out
        
    # Binning 9 pixels of the 8bit images
    # @staticmethod
    # @jit(nopython=True, fastmath=True, parallel=True, cache=True)
    # def bin9(arr_in):
    #     m,n,o   = np.shape(arr_in)
    #     arr_tmp = np.empty((m//9,n,o), dtype='uint16')
    #     arr_out = np.empty((m//9,n//9,o), dtype='uint16')
    #     for i in prange(m//9):
    #         arr_tmp[i,:,:] =  arr_in[i*9,:,:]   + arr_in[i*9+1,:,:]  + arr_in[i*9+2,:,:]  + arr_in[i*9+3,:,:]  +  arr_in[i*9+4,:,:] \
    #                        +  arr_in[i*9+5,:,:] + arr_in[i*9+6,:,:]  + arr_in[i*9+7,:,:]  + arr_in[i*9+8,:,:] 
    #     for j in prange(n//9):
    #         arr_out[:,j,:] = arr_tmp[:,j*9,:]   + arr_tmp[:,j*9+1,:] + arr_tmp[:,j*9+2,:] + arr_tmp[:,j*9+3,:] + arr_tmp[:,j*9+4,:] \
    #                        + arr_tmp[:,j*9+5,:] + arr_tmp[:,j*9+6,:] + arr_tmp[:,j*9+7,:] + arr_tmp[:,j*9+8,:]
    #     return arr_out
    @staticmethod
    @jit(nopython=True, fastmath=True, parallel=True, cache=True)
    def bin9(arr):
        m, n, o = arr.shape
        tmp = np.empty((m//9, n, o), dtype=np.uint16)
        out = np.empty((m//9, n//9, o), dtype=np.uint16)
        for i in prange(m//9):
            tmp[i] = (arr[9*i] + arr[9*i+1] + arr[9*i+2] + arr[9*i+3] + arr[9*i+4]
                      + arr[9*i+5] + arr[9*i+6] + arr[9*i+7] + arr[9*i+8])
        for j in prange(n//9):
            out[:, j] = (tmp[:, 9*j] + tmp[:, 9*j+1] + tmp[:, 9*j+2] + tmp[:, 9*j+3] + tmp[:, 9*j+4]
                         + tmp[:, 9*j+5] + tmp[:, 9*j+6] + tmp[:, 9*j+7] + tmp[:, 9*j+8])
        return out
    
    # Binning 10 pixels of the 8bit images
    # @staticmethod
    # @jit(nopython=True, fastmath=True, parallel=True, cache=True)
    # def bin10(arr_in):
    #     m,n,o   = np.shape(arr_in)
    #     arr_tmp = np.empty((m//10,n,o), dtype='uint16')
    #     arr_out = np.empty((m//10,n//10,o), dtype='uint16')
    #     for i in prange(m//10):
    #         arr_tmp[i,:,:] =  arr_in[i*10,:,:]   + arr_in[i*10+1,:,:] +  arr_in[i*10+2,:,:] +  arr_in[i*10+3,:,:] +  arr_in[i*10+4,:,:] \
    #                         + arr_in[i*10+5,:,:] + arr_in[i*10+6,:,:] +  arr_in[i*10+7,:,:] +  arr_in[i*10+8,:,:] +  arr_in[i*10+9,:,:]

    #     for j in prange(n//10):
    #         arr_out[:,j,:] = arr_tmp[:,j*10,:]   + arr_tmp[:,j*10+1,:] + arr_tmp[:,j*10+2,:] + arr_tmp[:,j*10+3,:] + arr_tmp[:,j*10+4,:] \
    #                        + arr_tmp[:,j*10+5,:] + arr_tmp[:,j*10+6,:] + arr_tmp[:,j*10+7,:] + arr_tmp[:,j*10+8,:] + arr_tmp[:,j*10+9,:]
    #     return arr_out
    @staticmethod
    @jit(nopython=True, fastmath=True, parallel=True, cache=True)
    def bin10(arr):
        m, n, o = arr.shape
        tmp = np.empty((m//10, n, o), dtype=np.uint16)
        out = np.empty((m//10, n//10, o), dtype=np.uint16)
        for i in prange(m//10):
            tmp[i] = (arr[10*i] + arr[10*i+1] + arr[10*i+2] + arr[10*i+3] + arr[10*i+4]
                      + arr[10*i+5] + arr[10*i+6] + arr[10*i+7] + arr[10*i+8] + arr[10*i+9])
        for j in prange(n//10):
            out[:, j] = (tmp[:, 10*j] + tmp[:, 10*j+1] + tmp[:, 10*j+2] + tmp[:, 10*j+3] + tmp[:, 10*j+4]
                         + tmp[:, 10*j+5] + tmp[:, 10*j+6] + tmp[:, 10*j+7] + tmp[:, 10*j+8] + tmp[:, 10*j+9])
        return out
    
    # Binning 12 pixels of the 8bit images
    # @staticmethod
    # @jit(nopython=True, fastmath=True, parallel=True, cache=True)
    # def bin12(arr_in):
    #     m,n,o   = np.shape(arr_in)
    #     arr_tmp = np.empty((m//12,n,o), dtype='uint16')
    #     arr_out = np.empty((m//12,n//12,o), dtype='uint32')
    #     for i in prange(m//12):
    #         arr_tmp[i,:,:] =  arr_in[i*12,:,:]    + arr_in[i*12+1,:,:]  + arr_in[i*12+2,:,:]  + arr_in[i*12+3,:,:]  + arr_in[i*12+4,:,:]  \
    #                         + arr_in[i*12+5,:,:]  + arr_in[i*12+6,:,:]  + arr_in[i*12+7,:,:]  + arr_in[i*12+8,:,:]  + arr_in[i*12+9,:,:]  \
    #                         + arr_in[i*12+10,:,:] + arr_in[i*12+11,:,:] 

    #     for j in prange(n//12):
    #         arr_out[:,j,:]  = arr_tmp[:,j*12,:]    + arr_tmp[:,j*12+1,:]  + arr_tmp[:,j*12+2,:] + arr_tmp[:,j*12+3,:] + arr_tmp[:,j*12+4,:] \
    #                         + arr_tmp[:,j*12+5,:]  + arr_tmp[:,j*12+6,:]  + arr_tmp[:,j*12+7,:] + arr_tmp[:,j*12+8,:] + arr_tmp[:,j*12+9,:] \
    #                         + arr_tmp[:,j*12+10,:] + arr_tmp[:,j*12+11,:] 
    #     return arr_out
    @staticmethod
    @jit(nopython=True, fastmath=True, parallel=True, cache=True)
    def bin12(arr):
        m, n, o = arr.shape
        tmp = np.empty((m//12, n, o), dtype=np.uint16)
        out = np.empty((m//12, n//12, o), dtype=np.uint32)
        for i in prange(m//12):
            s = 0
            for k in range(12):
                s += arr[12*i+k]
            tmp[i] = s
        for j in prange(n//12):
            t = 0
            for k in range(12):
                t += tmp[:, 12*j+k]
            out[:, j] = t
        return out
    
    # Binning 15 pixels of the 8bit images
    # @staticmethod
    # @jit(nopython=True, fastmath=True, parallel=True, cache=True)
    # def bin15(arr_in):
    #     m,n,o   = np.shape(arr_in)
    #     arr_tmp = np.empty((m//15,n,o), dtype='uint16')
    #     arr_out = np.empty((m//15,n//15,o), dtype='uint32')
    #     for i in prange(m//15):
    #         arr_tmp[i,:,:] =  arr_in[i*15,:,:]    + arr_in[i*15+1,:,:]  + arr_in[i*15+2,:,:]  + arr_in[i*15+3,:,:]  + arr_in[i*15+4,:,:]  \
    #                         + arr_in[i*15+5,:,:]  + arr_in[i*15+6,:,:]  + arr_in[i*15+7,:,:]  + arr_in[i*15+8,:,:]  + arr_in[i*15+9,:,:]  \
    #                         + arr_in[i*15+10,:,:] + arr_in[i*15+11,:,:] + arr_in[i*15+12,:,:] + arr_in[i*15+13,:,:] + arr_in[i*15+14,:,:] 

    #     for j in prange(n//15):
    #         arr_out[:,j,:]  = arr_tmp[:,j*15,:]    + arr_tmp[:,j*15+1,:]  + arr_tmp[:,j*15+2,:]  + arr_tmp[:,j*15+3,:]  + arr_tmp[:,j*15+4,:]  \
    #                         + arr_tmp[:,j*15+5,:]  + arr_tmp[:,j*15+6,:]  + arr_tmp[:,j*15+7,:]  + arr_tmp[:,j*15+8,:]  + arr_tmp[:,j*15+9,:]  \
    #                         + arr_tmp[:,j*15+10,:] + arr_tmp[:,j*15+11,:] + arr_tmp[:,j*15+12,:] + arr_tmp[:,j*15+13,:] + arr_tmp[:,j*15+14,:]
    #     return arr_out
    @staticmethod
    @jit(nopython=True, fastmath=True, parallel=True, cache=True)
    def bin15(arr):
        m, n, o = arr.shape
        tmp = np.empty((m//15, n, o), dtype=np.uint16)
        out = np.empty((m//15, n//15, o), dtype=np.uint32)
        for i in prange(m//15):
            s = 0
            for k in range(15):
                s += arr[15*i+k]
            tmp[i] = s
        for j in prange(n//15):
            t = 0
            for k in range(15):
                t += tmp[:, 15*j+k]
            out[:, j] = t
        return out
    
    # Binning 18 pixels of the 8bit images
    # @staticmethod
    # @jit(nopython=True, fastmath=True, parallel=True, cache=True)
    # def bin18(arr_in):
    #     m,n,o   = np.shape(arr_in)
    #     arr_tmp = np.empty((m//18,n,o), dtype='uint16')
    #     arr_out = np.empty((m//18,n//18,o), dtype='uint32')
    #     for i in prange(m//18):
    #         arr_tmp[i,:,:] =  arr_in[i*18,:,:]    + arr_in[i*18+1,:,:]  + arr_in[i*18+2,:,:]  + arr_in[i*18+3,:,:]  + arr_in[i*18+4,:,:]  \
    #                         + arr_in[i*18+5,:,:]  + arr_in[i*18+6,:,:]  + arr_in[i*18+7,:,:]  + arr_in[i*18+8,:,:]  + arr_in[i*18+9,:,:]  \
    #                         + arr_in[i*18+10,:,:] + arr_in[i*18+11,:,:] + arr_in[i*18+12,:,:] + arr_in[i*18+13,:,:] + arr_in[i*18+14,:,:] \
    #                         + arr_in[i*18+15,:,:] + arr_in[i*18+16,:,:] + arr_in[i*18+17,:,:] 

    #     for j in prange(n//18):
    #         arr_out[:,j,:]  = arr_tmp[:,j*18,:]    + arr_tmp[:,j*18+1,:]  + arr_tmp[:,j*18+2,:]  + arr_tmp[:,j*18+3,:]  + arr_tmp[:,j*18+4,:]  \
    #                         + arr_tmp[:,j*18+5,:]  + arr_tmp[:,j*18+6,:]  + arr_tmp[:,j*18+7,:]  + arr_tmp[:,j*18+8,:]  + arr_tmp[:,j*18+9,:]  \
    #                         + arr_tmp[:,j*18+10,:] + arr_tmp[:,j*18+11,:] + arr_tmp[:,j*18+12,:] + arr_tmp[:,j*18+13,:] + arr_tmp[:,j*18+14,:] \
    #                         + arr_tmp[:,j*18+15,:] + arr_tmp[:,j*18+16,:] + arr_tmp[:,j*18+17,:]  
    #     return arr_out
    @staticmethod
    @jit(nopython=True, fastmath=True, parallel=True, cache=True)
    def bin18(arr):
        m, n, o = arr.shape
        tmp = np.empty((m//18, n, o), dtype=np.uint16)
        out = np.empty((m//18, n//18, o), dtype=np.uint32)
        for i in prange(m//18):
            s = 0
            for k in range(18):
                s += arr[18*i+k]
            tmp[i] = s
        for j in prange(n//18):
            t = 0
            for k in range(18):
                t += tmp[:, 18*j+k]
            out[:, j] = t
        return out

    # Binning 20 pixels of the 8bit images
    # @staticmethod
    # @jit(nopython=True, fastmath=True, parallel=True, cache=True)
    # def bin20(arr_in):
    #     m,n,o   = np.shape(arr_in)
    #     arr_tmp = np.empty((m//20,n,o), dtype='uint16')
    #     arr_out = np.empty((m//20,n//20,o), dtype='uint32')
    #     for i in prange(m//20):
    #         arr_tmp[i,:,:] =  arr_in[i*20,:,:]  + arr_in[i*20+1,:,:]  + arr_in[i*20+2,:,:]  + arr_in[i*20+3,:,:]  + arr_in[i*20+4,:,:]  + arr_in[i*20+5,:,:]  + \
    #                         arr_in[i*20+6,:,:]  + arr_in[i*20+7,:,:]  + arr_in[i*20+8,:,:]  + arr_in[i*20+9,:,:]  + arr_in[i*20+10,:,:] + arr_in[i*20+11,:,:] + \
    #                         arr_in[i*20+12,:,:] + arr_in[i*20+13,:,:] + arr_in[i*20+14,:,:] + arr_in[i*20+15,:,:] + arr_in[i*20+16,:,:] + arr_in[i*20+17,:,:] + \
    #                         arr_in[i*20+18,:,:] + arr_in[i*20+19,:,:]

    #     for j in prange(n//20):
    #         arr_out[:,j,:]  = arr_tmp[:,j*20,:]  + arr_tmp[:,j*20+1,:]  + arr_tmp[:,j*20+2,:]  + arr_tmp[:,j*20+3,:]  + arr_tmp[:,j*10+4,:]  + arr_tmp[:,j*20+5,:]  + \
    #                         arr_tmp[:,j*20+6,:]  + arr_tmp[:,j*20+7,:]  + arr_tmp[:,j*20+8,:]  + arr_tmp[:,j*20+9,:]  + arr_tmp[:,j*20+10,:] + arr_tmp[:,j*20+11,:] + \
    #                         arr_tmp[:,j*20+12,:] + arr_tmp[:,j*20+13,:] + arr_tmp[:,j*10+14,:] + arr_tmp[:,j*20+15,:] + arr_tmp[:,j*20+16,:] + arr_tmp[:,j*20+17,:] + \
    #                         arr_tmp[:,j*20+18,:] + arr_tmp[:,j*20+19,:] 
    #     return arr_out
    @staticmethod
    @jit(nopython=True, fastmath=True, parallel=True, cache=True)
    def bin20(arr):
        m, n, o = arr.shape
        tmp = np.empty((m//20, n, o), dtype=np.uint16)
        out = np.empty((m//20, n//20, o), dtype=np.uint32)
        for i in prange(m//20):
            s = 0
            for k in range(20):
                s += arr[20*i+k]
            tmp[i] = s
        for j in prange(n//20):
            t = 0
            for k in range(20):
                t += tmp[:, 20*j+k]
            out[:, j] = t
        return out

    def cube2DisplayImage(self, displayImage, indx=[0], name=[]):
        """ 
        Flattens the data cube to a display image.
        If 3 channels are selected, this requires 2x2 tile. 
        It will add channel label to the image tiles. 
        indx is selected channels
        name is the channel names with same length as indx
        """
        
        font             = cv2.FONT_HERSHEY_SIMPLEX
        fontScale        = 1
        lineType         = 2
        
        (depth,height,width) = self.data.shape
        # if len(indx) == depth:
        # maybe faster option if all images are selected
        # 
        
        # arrange selected images in a grid
        columns = math.ceil(math.sqrt(len(indx))) # how many columns are needed?
        rows    = math.ceil(len(indx)/columns)     # how many rows are needed?
        empty   = np.zeros((height,width), dtype=_htmp.dtype)
        i = 0
        for y in range(rows):
            _htmp = self.data[indx[i],:,:]
            for x in range(columns-1):
                if i < len(indx):
                    _htmp=cv2.hconcat((_htmp,self.data[indx[i+1],:,:]))
                else:
                    _htmp=cv2.hconcat((_htmp,empty))
                i +=1
            if y == 0:
                _vtmp = _htmp
            else:
                _vtmp=cv2.vconcat((_vtmp,_htmp))

        # resize grid of images to fit into display image
        (height, width) = _vtmp.shape[:2]
        (newHeight, newWidth) = displayImage.shape[:2]
        scale = max(height/newHeight, width/newWidth)
        dsize = (int(width // scale), int(height // scale))
        img = cv2.resize(_vtmp, dsize, cv2.INTER_LINEAR)
        (height, width) = img.shape[:2]
        # copy grid image into display image: display image has 3 channels        
        displayImage[0:height,0:width,0] = img
        displayImage[0:height,0:width,1] = img
        displayImage[0:height,0:width,2] = img
        if len(name) > 0:
            # add text to the individual images
            x = y = 0
            dx = width / columns
            dy = height / rows
            for i in indx:
                (Label_width, Label_height), BaseLine = cv2.getTextSize(name[i], fontFace=font, fontScale=fontScale, thickness=lineType)
                h = Label_height + BaseLine
                loc_x = int(x * dx)
                loc_y = int(y * dy + h)
                cv2.rectangle(displayImage,
                    (loc_x, loc_y),
                    (loc_x + Label_width, loc_y + h),
                    (255,255,255), -1, )
                cv2.putText(displayImage, 
                    text=name[i], 
                    org=(loc_x, loc_y), 
                    fontFace=font, 
                    fontScale=fontScale, 
                    color=(0,0,0), 
                    thickness=lineType)
                # update location
                x += 1
                if x >= columns-1: 
                    x=0
                    y+=1


class QDataDisplay(QObject):

    # Transform band passed data to display image
    # Goal is to enhance small changes and to convert data to 0..1 range
    # A few example options:
    # data = np.sqrt(np.multiply(data,abs(data_bandpass)))
    # data = np.sqrt(255.*np.absolute(data_highpass)).astype('uint8')
    # data = (128.-data_highpass).astype('uint8')
    # data = np.left_shift(np.sqrt((np.multiply(data_lowpass,np.absolute(data_highpass)))).astype('uint8'),2)
    @staticmethod
    @vectorize(['float32(float32)'], nopython=True, fastmath=True, cache=True)
    def displaytrans(val:np.float32):
        return np.sqrt(16.*np.abs(val))

    @staticmethod
    def resize_img(
        img: np.ndarray,
        target_width: int,
        target_height: int,
        pad: bool = False,
        border_type: int = cv2.BORDER_CONSTANT,
        pad_value: int = 0
    ) -> tuple:
        """
        Resize `img` to fit within (target_width, target_height) while
        preserving aspect ratio. If `pad` is True, center-pad to exact
        dimensions using `border_type` and `pad_value`.

        Returns:
            resized: np.ndarray  # the resized (and optionally padded) image
            scale: float         # scaling factor applied
            left: int            # left padding
            top: int             # top padding
        """

        # origin: UU
        
        # We can  also place img on top left corner and pad with black pixels on the right and bottom of the image
        #   return np.pad(img_r, ((t, b), (l, r), (0,0)), mode=mode), factor    
        #   return np.pad(img_r, ((0, diff_y), (0, diff_x), (0,0)), mode=mode), factor 
        #   copyMakeBorder seems to be faster
        
        h, w = img.shape[:2]
        scale = min(target_width / w, target_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        if not pad:
            return resized, scale, 0, 0

        delta_w = target_width - new_w
        delta_h = target_height - new_h
        left = delta_w // 2
        top = delta_h // 2
        right = delta_w - left
        bottom = delta_h - top

        # use copyMakeBorder for speed
        if resized.ndim == 2:
            padded = cv2.copyMakeBorder(
                resized, top, bottom, left, right,
                borderType=border_type, value=pad_value
            )
        else:
            padded = cv2.copyMakeBorder(
                resized, top, bottom, left, right,
                borderType=border_type, value=[pad_value]*3
            )
        return padded, scale, left, top

    def compose_grid(
            data: np.ndarray,
            indices: list,
            labels: list = None,
            grid_size: tuple = None
        ) -> np.ndarray:        
        """ 
        Tile selected slices from a 3D data array into a grid.

        Args:
            data: np.ndarray        # shape (depth, H, W)
            indices: List[int]      # slices to include
            labels: List[str]       # optional text labels
            grid_size: (rows,cols)  # fixed grid shape, overrides auto-compute

        Returns:
            grid_img: np.ndarray    # single-channel grid image            
        """
        
        depth, H, W = data.shape
        n = len(indices)
        if grid_size:
            rows, cols = grid_size
        else:
            cols = math.ceil(math.sqrt(n))
            rows = math.ceil(n / cols)

        # prepare blank tile
        blank = np.zeros((H, W), dtype=data.dtype)
        # pad indices list to fill grid
        padded = indices + [-1] * (rows*cols - n)

        # build tiles row by row
        rows_img = []
        for r in range(rows):
            tiles = []
            for c in range(cols):
                idx = padded[r*cols + c]
                tile = blank if (idx < 0 or idx >= depth) else data[idx]
                tiles.append(tile)
            row_img = cv2.hconcat(tiles)
            rows_img.append(row_img)
        grid = cv2.vconcat(rows_img)

        # overlay labels if provided
        if labels:
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.5
            thickness = 1
            for i, lbl in enumerate(labels[:n]):
                r = i // cols
                c = i % cols
                x = c * W + 5
                y = r * H + int(20 * scale)
                cv2.putText(grid, lbl, (x, y), font, scale, color=255, thickness=thickness)

        return grid
    
class threeBandEqualizerProcessor():
    """3 Band Equalizer"""

    # Initialize the Processor Thread
    def __init__(self, res: tuple, gain_low: float, gain_mid: float, gain_high: float, fc_low: float, fc_high: float, fs: float):

        self.fs  = fs
        self.fcl = fc_low
        self.fch = fc_high
     
             # Gain Controls
        self.lg   = gain_low     # low  gain
        self.mg   = gain_mid     # mid  gain
        self.hg   = gain_high    # high gain
           
        # Filter #1 (Low band)   
        self.f1p0 = np.zeros(res, 'float32')   # Poles ...
        self.f1p1 = np.zeros(res, 'float32')
        self.f1p2 = np.zeros(res, 'float32')
        self.f1p3 = np.zeros(res, 'float32')

        # Filter #2 (High band)
        self.f2p0 = np.zeros(res, 'float32')   # Poles ...
        self.f2p1 = np.zeros(res, 'float32')
        self.f2p2 = np.zeros(res, 'float32')
        self.f2p3 = np.zeros(res, 'float32')

        # Sample history buffer
        self.sdm1 = np.zeros(res, 'float32')   # Sample data minus 1
        self.sdm2 = np.zeros(res, 'float32')   #                   2
        self.sdm3 = np.zeros(res, 'float32')   #                   3

        self.vsa = 1.0 / 4294967295.0          # Very small amount (Denormal Fix)

        self.lf = 2 * math.sin(math.pi * (self.fcl / self.fs))
        self.hf = 2 * math.sin(math.pi * (self.fch / self.fs))

        self.l = np.zeros(res, 'float32')      # low  frequency sample
        self.m = np.zeros(res, 'float32')      # mid  frequency sample
        self.h = np.zeros(res, 'float32')      # high frequency sample
  
    def equalize(self, data):
        """ three band equalizer """
       
        start_time = time.perf_counter()

        # Filter #1 (low pass)
        self.f1p0  += (self.lf * (data      - self.f1p0)) + self.vsa
        self.f1p1  += (self.lf * (self.f1p0 - self.f1p1))
        self.f1p2  += (self.lf * (self.f1p1 - self.f1p2))
        self.f1p3  += (self.lf * (self.f1p2 - self.f1p3))
        self.l      = self.f1p3

        # Filter #2 (high pass)
        self.f2p0  += (self.hf * (data      - self.f2p0)) + self.vsa
        self.f2p1  += (self.hf * (self.f2p0 - self.f2p1))
        self.f2p2  += (self.hf * (self.f2p1 - self.f2p2))
        self.f2p3  += (self.hf * (self.f2p2 - self.f2p3))
        self.h      = self.sdm3 - self.f2p3

        # Calculate midrange (signal - (low + high))
        self.m      = self.sdm3 - (self.h + self.l)
        
        # Scale, combine and return signal
        self.l     *= self.lg
        self.m     *= self.mg
        self.h     *= self.hg

        # Shuffle history buffer
        self.sdm3   = self.sdm2
        self.sdm2   = self.sdm1
        self.sdm1   = data

        total_time += time.perf_counter() - start_time

        return self.l + self.lm + self.lh

class ThreeBandEqualizerProcessor_scipy:
    """
    Optimized 3-band audio/image equalizer using SciPy SOS filters.

    Applies low, mid, and high gain to a streaming 2D signal (e.g., image stack or audio frames)
    with minimal Python overhead by leveraging SciPy's C-optimized filter routines.

    Attributes:
        sos_low: ndarray       # second-order sections for low-pass
        sos_high: ndarray      # second-order sections for high-pass
        gain_low: float
        gain_mid: float
        gain_high: float
    """
    def __init__(
        self,
        cutoff_low: float,
        cutoff_high: float,
        fs: float,
        gain_low: float = 1.0,
        gain_mid: float = 1.0,
        gain_high: float = 1.0,
        order: int = 4
    ):
        # design 4th-order Butterworth filters as SOS
        self.sos_low = butter(order, cutoff_low/(fs/2), btype='low', output='sos')
        self.sos_high = butter(order, cutoff_high/(fs/2), btype='high', output='sos')

        self.gain_low = gain_low
        self.gain_mid = gain_mid
        self.gain_high = gain_high

    def equalize(self, data: np.ndarray) -> np.ndarray:
        """
        Process one frame or batch of frames: apply low-, high-, and mid-band gains.

        Args:
            data: np.ndarray  # input signal, shape (..., N) or (..., H, W)

        Returns:
            out: np.ndarray   # equalized output, same shape as `data`
        """
        # low-frequency component
        low = sosfilt(self.sos_low, data, axis=-1)
        # high-frequency component
        high = sosfilt(self.sos_high, data, axis=-1)
        # mid-frequency is residual
        mid = data - low - high

        # combine with gains
        return (self.gain_low * low
                + self.gain_mid * mid
                + self.gain_high * high)

###############################################################################
# High Pass Image Processor
# Poor Man's
###############################################################################
# Construct poor man's low pass filter y = (1-alpha) * y + alpha * x
# https://dsp.stackexchange.com/questions/54086/single-pole-iir-low-pass-filter-which-is-the-correct-formula-for-the-decay-coe
# f_s sampling frequency in Hz
# f_c cut off frequency in Hz
# w_c radians 0..pi (pi is Niquist fs/2, 2pi is fs)
###############################################################################
# Urs Utzinger 2022

# class poormansHighpassProcessor():
#     """
#     Highpass filter
#     y = (1-alpha) * y + alpha * x
#     """

#     # Initialize 
#     def __init__(self, res: tuple = (14,720,540), alpha: float = 0.95 ):

#         # Initialize Processor
#         self.alpha = alpha
#         self.averageData  = np.zeros(res, 'float32')
#         self.filteredData  = np.zeros(res, 'float32')

#     # After Starting the Thread, this runs continuously
#     def highpass(self, data):
#         start_time = time.perf_counter()
#         self.averageData  = self.movingavg(data, self.averageData, self.alpha)
#         self.filteredData = self.highpass(data, self.averageData)
#         total_time += time.perf_counter() - start_time
#         return self.filteredData

#     # Numpy Vectorized Image Processor
#     # y = (1-alpha) * y + alpha * x
#     @vectorize(['float32(uint16, float32, float32)'], nopython=True, fastmath=True)
#     def movingavg(data, average, alpha):
#         return np.add(np.multiply(average, 1.-alpha), np.multiply(data, alpha))

#     @vectorize(['float32(uint16, float32)'], nopython=True, fastmath=True)
#     def highpass(data, average):
#         return np.subtract(data, average)

#     def computeAlpha(f_s = 50.0, f_c = 5.0):
#         w_c = (2.*3.141) * f_c / f_s         # normalized cut off frequency [radians]
#         y = 1 - math.cos(w_c);               # compute alpha for 3dB attenuation at cut off frequency
#         # y = w_c*w_c / 2.                   # small angle approximation
#         return -y + math.sqrt( y*y + 2.*y ); # 


###############################################################################
# Highpass Filter
# Running Summ
#
# Moving Average (D-1 additions per sample)
# y(n) =  Sum(x(n-i))i=1..D * 1/D 
#
# Recursive Running Sum (one addition and one subtraction per sample)
# y(n) = [ x(n) - x(n-D) ] * 1/D + y(n-1)
#
# Cascade Integrator Comb Filter as Moving Average Filter
# y(n) = ( x(n) - x(n-D) ) + y(n-1)
#
# https://en.wikipedia.org/wiki/Cascaded_integrator-comb_filter
# https://dsp.stackexchange.com/questions/12757/a-better-high-order-low-pass-filter
# https://www.dsprelated.com/freebooks/sasp/Running_Sum_Lowpass_Filter.html
# https://www.dsprelated.com/showarticle/1337.php
###############################################################################
# Urs Utzinger 2022

# class runningsumHighpassProcessor(QThread):
#     """Highpass filter"""

#     # Initialize the Processor Thread
#     def __init__(self, res: tuple = (14, 720, 540), delay: int = 1 ):

#         # Initialize Processor
#         self.data_lowpass  = np.zeros(res, 'float32')
#         self.data_highpass = np.zeros(res, 'float32')

#         self.circular_buffer = self.collections.deque(maxlen=delay)
#         # initialize buffer with zeros
#         for i in range(delay):
#             self.circular_buffer.append(self.data_lowpass)

#     @vectorize(['uint8(uint8, uint8, uint8)'], nopython=True, fastmath=True)
#     def runsum(data, data_delayed, data_previous):
#         # Numpy Vectorized Image Processor
#         # y(n) = ( x(n) - x(n-D) ) + y(n-1)
#         # x(n), x(n-D) y(n-1)
#         return np.add(np.subtract(data, data_delayed), data_previous)

#     @vectorize(['uint8(uint8, uint8)'], nopython=True, fastmath=True)
#     def highpass(data, data_filtered):
#         return np.subtract(data, data_filtered)

#     def highpass(self, data):
#         start_time = time.perf_counter()
#         xn = data                                     # x(n)
#         xnd = self.circular_buffer.popleft()          # x(N-D)
#         self.circular_buffer.append(data)             # put new data into delay line
#         yn1 = self.data_lowpass                       # y(n-1)
#         self.data_lowpass = self.runsum(xn, xnd, yn1)      # y(n) = x(n) - x(n-D) + y(n-1)
#         self.data_hihgpass = self.highpass(data, self.data_lowpass)
#         total_time += time.perf_counter() - start_time

# Is same as

class OnePoleHighpass:
    # alpha = exp(-2πfc/fs
    def __init__(self, alpha):
        # b = [1, -1], a = [1, -alpha]
        self.b = [1.0, -1.0]
        self.a = [1.0, -alpha]
        self.zi = None

    def highpass(self, x):
        y, self.zi = lfilter(self.b, self.a, x, axis=-1, zi=self.zi)
        return y