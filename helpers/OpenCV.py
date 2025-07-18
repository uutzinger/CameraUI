###########################################################################################
# OpenCV Camera Class
###########################################################################################

# Open CV Camera Driver
import cv2
from threading import Thread,Lock
# System
import logging, time,sys
# Numerical Tools
import numpy as np

# QT
try:
    from PyQt6.QtCore import QObject, QTimer, QThread, pyqtSignal, pyqtSlot, QSignalMapper, QCoreApplication
    from PyQt6.QtWidgets import QLineEdit, QSlider, QCheckBox, QLabel
    hastQt6 = True
except:
    from PyQt5.QtCore import QObject, QTimer, QThread, pyqtSignal, pyqtSlot, QSignalMapper, QCoreApplication
    from PyQt5.QtWidgets import QLineEdit, QSlider, QCheckBox, QLabel
    hastQt6 = False

from helpers.Processing_helper import QDataCube

class OpenCVCapture(QObject):

    imageDataReady      = pyqtSignal(float, np.ndarray)                                 # image received on serial port
    fpsReady            = pyqtSignal(float)                                             # FPS ready signal          
    logSignal           = pyqtSignal(int, str)                                          # Logging

    # Initialize the Camera Thread
    # Opens Capture Device and Sets Capture Properties
    ############################################################################
    def __init__(self, configs, 
        camera_num: int = 0,             # more than one camera?
        res: tuple = None,               # width, height
        exposure: float = None           # exposure over write
        ):

        super().__init__()

        # populate desired settings from configuration file or function arguments
        ####################################################################
        self._camera_num       = camera_num
        self.logger = logging.getLogger("OpenCV.py")
        if exposure is not None:
            self._exposure    = exposure  
        else: 
            if 'exposure' in configs: self._exposure       = configs['exposure']
            else:                     self._exposure       = -1.0
        if res is not None:
            self._camera_res = res
        else: 
            if 'camera_res' in configs: self._camera_res   = configs['camera_res']
            else:                       self._camera_res   = (640, 480)
        if 'fps' in configs:            self._framerate    = configs['fps']
        else:                           self._framerate    = -1.0
        if 'buffersize' in configs:     self._buffersize   = configs['buffersize']         # camera drive buffer size
        else:                           self._buffersize   = -1
        if 'fourcc' in configs:         self._fourcc       = configs['fourcc']             # camera sensor encoding format
        else:                           self._fourcc       = -1
        if 'autoexposure' in configs:   self._autoexposure = configs['autoexposure']       # autoexposure depends on camera
        else:                           self._autoexposure = -1
        if 'gain' in configs:           self._gain         = configs['gain']
        else:                           self._gain         = -1.0
        if 'wb_temp' in configs:        self._wbtemp       = configs['wb_temp']
        else:                           self._wbtemp       = -1
        if 'autowb' in configs:         self._autowb       = configs['autowb']
        else:                           self._autowb       = -1
        if 'settings' in configs:       self._settings     = configs['settings']
        else:                           self._settings     = -1
        
        # Init vars
        self.measured_fps = 0.0
        self.stopped            = True
        self.camera_lock        = Lock()
        self.last_time_read     = time.perf_counter() 
        self.last_fps_emit      = time.perf_counter()

    @pyqtSlot
    def on_setupCameraRequest(self):

        self.thread_id = int(QThread.currentThreadId()) if QThread.currentThreadId() else "N/A"

        self._openCamera()

        # setup the throughput measurement timer
        self.throughputTimer = QTimer()
        self.throughputTimer.setInterval(1000)
        self.throughputTimer.timeout.connect(self.on_throughputTimer)
        self.handle_log(
            logging.INFO,
            f"[{self.thread_id}]: setup fps timer."
        )

        self.handle_log(
            logging.INFO,
            f"[{self.thread_id}]: Camera setup."
        )

    @pyqtSlot
    def on_closeCameraRequest(self):
        try: self.camera.release()
        except: pass

    @pyqtSlot
    def on_startAcquisitionRequest(self, depth=1):
        # create datacube structure       
        self.datacube = QDataCube(width=self.width, height=self.height, depth=depth)
        self.datacube.dataCubeReady.connect(self.imageDataReady.emit)
        self.stopped = False
        self.handle_log(logging.INFO, "[OpenCV]: Acquiring images.")
        while not self.stopped:
            self._read()
            QCoreApplication.processEvents() # head to other events

    @pyqtSlot
    def on_stopAcquisitionRequest(self):
        self.stopped = True
        del self.datacube
        self.handle_log(logging.INFO, "[OpenCV]: Stopped acquiring images.")

    @pyqtSlot()
    def on_throughputTimer(self) -> None:
        """
        Report fps
        """
        if self.camera_open:
            self.fpsReady.emit(self.measured_fps)
        else:
            self.fpsReady.emit(0.0)

    ########################################################################################
    # Internal Camera Functions
    ########################################################################################

    def _read(self):
        """
        Read Capture Device
        """
       
        current_time = time.perf_counter()
      
        if self.camera is None:
            return
        
        raw = None
        with self.camera_lock:
            if not self.camera.grab():
                return
            ok, frame = self.camera.retrieve() # should always have frame as we just checked with grab
            if not ok:
                return
            raw = frame.copy()
            
        if raw.ndim>=3):
            img = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        else:
            img = raw
 
        if img is None:
            self.handle_log(logging.WARNING, "[CAM]:no image available!")
            return

        self.datacube.add(img)

        # FPS calculation
        dt = current_time - self.last_time_read
        if dt > 0:
            self.measured_fps = (0.9 * self.measured_fps) + (0.1/dt)
        self.last_time_read = current_time
        
    def _openCamera(self):
        """
        Open up the camera so we can begin capturing frames
        """

        # Open the camera with platform optimal settings
        if sys.platform.startswith('win'):
            self.camera = cv2.VideoCapture(self._camera_num, apiPreference=cv2.CAP_DSHOW) # CAP_VFW or CAP_DSHOW or CAP_MSMF or CAP_ANY
        elif sys.platform.startswith('darwin'):
            self.camera = cv2.VideoCapture(self._camera_num, apiPreference=cv2.CAP_AVFOUNDATION)
        elif sys.platform.startswith('linux'):
            self.camera = cv2.VideoCapture(self._camera_num, apiPreference=cv2.CAP_V4L2)
        else:
            self.camera = cv2.VideoCapture(self._camera_num, apiPreference=cv2.CAP_ANY)

        self.camera_open = self.camera.isOpened()

        if self.camera_open:
            # Apply settings to camera
            #self.height        = self._camera_res[1]   # image resolution
            #self.width         = self._camera_res[0]   # image resolution
            self.resolution    = self._camera_res      #
            self.exposure      = self._exposure        # camera exposure
            self.autoexposure  = self._autoexposure    # autoexposure
            self.fps           = self._framerate       # desired fps
            self.buffersize    = self._buffersize      # camera drive buffer size
            self.fourcc        = self._fourcc          # camera sensor encoding format
            self.gain          = self._gain            # camera gain
            self.wbtemperature = self._wbtemp          # camera white balance temperature
            self.autowb        = self._autowb          # camera enable auto white balance

            if self._settings > -1: self.camera.set(cv2.CAP_PROP_SETTINGS, 0.0) # open camera settings window
            
            # Update records
            self._camera_res    = self.resolution
            self._exposure      = self.exposure
            self._buffersize    = self.buffersize
            self._framerate     = self.fps
            self._autoexposure  = self.autoexposure
            self._fourcc        = self.fourcc
            self._fourcc_str    = self.decode_fourcc(self._fourcc)
            self._gain          = self.gain
            self._wbtemperature = self.wbtemperature
            self._autowb        = self.autowb
        else:
            self.handle_log(logging.CRITICAL, "[OpenCV]: Failed to open camera!")

    ########################################################################################
    # Utility Functions
    ########################################################################################


    def handle_log(self, level: int, message:str) -> None:
        """Emit the log signal with a level and message."""
        self.logSignal.emit(level, message)

    def opensettings(self):
        """
        Open up the camera settings window
        """
        if self.camera_open:
            self.camera.set(cv2.CAP_PROP_SETTINGS, 0.0)

    ########################################################################################
    # Camera Properties
    #   Reading and Setting
    ########################################################################################

    @property
    def width(self):
        """ returns video capture width """
        if self.camera_open:
            return int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        else: return -1
    @width.setter
    def width(self, val):
        """ sets video capture width """
        if (val is None) or (val == -1):
            self.handle_log(logging.WARNING, "[OpenCV]: Width not changed to {}.".format(val))
            return
        if self.camera_open and val > 0:
            with self.camera_lock:
                if self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, val):
                    # self._camera_res = (int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self._camera_res[1]))
                    # HEIGHT and WIDTH only valid if both were set
                   self.handle_log(logging.INFO, "[OpenCV]: Width:{}.".format(val))
                else:
                    self.handle_log(logging.ERROR, "[OpenCV]: Failed to set Width to {}.".format(val))
        else:
            self.handle_log(logging.CRITICAL, "[OpenCV]: Failed to set Width, camera not open!")

    @property
    def height(self):
        """ returns video capture height """
        if self.camera_open:
            return int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        else: return -1
    @height.setter
    def height(self, val):
        """ sets video capture height """
        if (val is None) or (val == -1):
            self.handle_log(logging.WARNING, "[OpenCV]: Height not changed:{}.".format(val))
            return
        if self.camera_open and val > 0:
            with self.camera_lock:
                if self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, int(val)):
                    # self._camera_res = (int(self._camera_res[0]), int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                    # HEIGHT and WIDTH only valid if both were set
                    self.handle_log(logging.INFO, "[OpenCV]: Height:{}.".format(val))
                else:
                    self.handle_log(logging.ERROR, "[OpenCV]: Failed to set Height to {}.".format(val))
        else:
            self.handle_log(logging.CRITICAL, "[OpenCV]: Failed to set Height, camera not open!")

    @property
    def resolution(self):
        """ returns current resolution width x height """
        if self.camera_open:
            return (int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                    int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        else: return (-1, -1) 
    @resolution.setter
    def resolution(self, val):
        if val is None: return
        if self.camera_open:
            if len(val) > 1: # have width x height
                self.width  = int(val[0])
                self.height = int(val[1])
            else: # given only one value for resolution
                self.width  = int(val)
                self.height = int(val)
            self._camera_res = (int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self.handle_log(logging.INFO, "[OpenCV]: Resolution:{}x{}.".format(self._camera_res[0],self._camera_res[1]))
        else: # camera not open
            self.handle_log(logging.CRITICAL, "[OpenCV]: Failed to set Resolution, camera not open!")

    @property
    def exposure(self):
        """ returns current exposure """
        if self.camera_open:
            return self.camera.get(cv2.CAP_PROP_EXPOSURE)
        else: return float("NaN")
    @exposure.setter
    def exposure(self, val):
        """ # sets current exposure """
        if (val is None):
            self.handle_log(logging.WARNING, "[OpenCV]: Skipping set Exposure to {}.".format(val))
            return
        if self.camera_open:
            with self.camera_lock:
                if self.camera.set(cv2.CAP_PROP_EXPOSURE, val):
                    self.handle_log(logging.INFO, "[OpenCV]: Exposure set:{}.".format(val))
                    self._exposure = self.camera.get(cv2.CAP_PROP_EXPOSURE)
                    self.handle_log(logging.INFO, "[OpenCV]: Exposure is:{}.".format(self._exposure))
                else:
                    self.handle_log(logging.ERROR, "[OpenCV]: Failed to set Expsosure to:{}.".format(val))
        else:
            self.handle_log(logging.CRITICAL, "[OpenCV]: Failed to set Exposure, camera not open!")

    @property
    def autoexposure(self):
        """ returns current exposure """
        if self.camera_open:
            return int(self.camera.get(cv2.CAP_PROP_AUTO_EXPOSURE))
        else: return -1
    @autoexposure.setter
    def autoexposure(self, val):
        """ sets autoexposure """
        if (val is None):
            self.handle_log(logging.WARNING, "[OpenCV]: Skipping set Autoexposure to:{}.".format(val))
            return
        if self.camera_open:
            with self.camera_lock:
                if self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, val):
                    self.handle_log(logging.INFO, "[OpenCV]: Autoexposure set:{}.".format(val))
                    self._autoexposure = self.camera.get(cv2.CAP_PROP_AUTO_EXPOSURE)
                    self.handle_log(logging.INFO, "[OpenCV]: Autoexposure is:{}.".format(self._autoexposure))
                else:
                    self.handle_log(logging.ERROR, "[OpenCV]: Failed to set Autoexposure to:{}.".format(val))
        else:
            self.handle_log(logging.CRITICAL, "[OpenCV]: Failed to set Autoexposure, camera not open!")

    @property
    def fps(self):
        """ returns current frames per second setting """
        if self.camera_open:
            return self.camera.get(cv2.CAP_PROP_FPS)
        else: return float("NaN")
    @fps.setter
    def fps(self, val):
        """ set frames per second in camera """
        if (val is None) or (val == -1):
            self.handle_log(logging.WARNING, "[OpenCV]: Skipping set FPS to:{}.".format(val))
            return
        if self.camera_open:
            with self.camera_lock:
                if self.camera.set(cv2.CAP_PROP_FPS, val):
                    self.handle_log(logging.INFO, "[OpenCV]: FPS set:{}.".format(val))
                    self._framerate = self.camera.get(cv2.CAP_PROP_FPS)
                    self.handle_log(logging.INFO, "[OpenCV]: FPS is:{}.".format(self._framerate))
                else:
                    self.handle_log(logging.ERROR, "[OpenCV]: Failed to set FPS to:{}.".format(val))
        else:
            self.handle_log(logging.CRITICAL, "[OpenCV]: Failed to set FPS, camera not open!")

    @staticmethod
    def decode_fourcc(val):
        """ decode the fourcc integer to the character string """
        return "".join([chr((int(val) >> 8 * i) & 0xFF) for i in range(4)])

    @property
    def fourcc(self):
        """ return video encoding format """
        if self.camera_open:
            return int(self.camera.get(cv2.CAP_PROP_FOURCC))
        else: return "None"
    @fourcc.setter
    def fourcc(self, val):
        """ set video encoding format in camera """
        if (val is None) or (val == -1):
            self.handle_log(logging.WARNING, "[OpenCV]: Skipping set FOURCC to:{}.".format(val))
            return
        if self.camera_open:        
            if isinstance(val, str): # fourcc is a string
                with self.camera_lock: 
                    if self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(val[0],val[1],val[2],val[3])):
                        self._fourcc     = self.camera.get(cv2.CAP_PROP_FOURCC)
                        self._fourcc_str = self.decode_fourcc(self._fourcc)
                        self.handle_log(logging.INFO, "[OpenCV]: FOURCC is:{}.".format(self._fourcc_str))
                    else:
                        self.handle_log(logging.ERROR, "[OpenCV]: Failed to set FOURCC to:{}.".format(val))
            else: # fourcc is integer/long
                with self.camera_lock: 
                    if self.camera.set(cv2.CAP_PROP_FOURCC, val):
                        self._fourcc     = int(self.camera.get(cv2.CAP_PROP_FOURCC))
                        self._fourcc_str = self.decode_fourcc(self._fourcc)
                        self.handle_log(logging.INFO, "[OpenCV]: FOURCC is:{}.".format(self._fourcc_str))
                    else:
                        self.handle_log(logging.ERROR, "[OpenCV]: Failed to set FOURCC to:{}.".format(val))
        else:
            self.handle_log(logging.CRITICAL, "[OpenCV]: Failed to set fourcc, camera not open!")

    @property
    def buffersize(self):
        """ return opencv camera buffersize """
        if self.camera_open:
            return int(self.camera.get(cv2.CAP_PROP_BUFFERSIZE))
        else: return float("NaN")
    @buffersize.setter
    def buffersize(self, val):
        """ set opencv camera buffersize """
        if val is None or val < 0:
            self.handle_log(logging.WARNING, "[OpenCV]: Skipping set Buffersize to:{}.".format(val))
            return
        if self.camera_open:
            with self.camera_lock:
                if self.camera.set(cv2.CAP_PROP_BUFFERSIZE, val):
                    self.handle_log(logging.INFO, "[OpenCV]: Buffersize set:{}.".format(val))
                    self._buffersize = int(self.camera.get(cv2.CAP_PROP_BUFFERSIZE))
                    self.handle_log(logging.INFO, "[OpenCV]: Buffersize is:{}.".format(self._buffersize))
                else:
                    self.handle_log(logging.ERROR, "[OpenCV]: Failed to set Buffersize to:{}.".format(val))
        else:
            self.handle_log(logging.CRITICAL, "[OpenCV]: Failed to set Buffersize, camera not open!")

    @property
    def gain(self):
        """ return opencv camera gain """
        if self.camera_open:
            return int(self.camera.get(cv2.CAP_PROP_GAIN))
        else: return float("NaN")
    @gain.setter
    def gain(self, val):
        """ set opencv camera gain """
        if val is None or val < 0:
            self.handle_log(logging.WARNING, "[OpenCV]: Skipping set Gain to:{}.".format(val))
            return
        if self.camera_open:
            with self.camera_lock:
                if self.camera.set(cv2.CAP_PROP_GAIN, val):
                    self.handle_log(logging.INFO, "[OpenCV]: Gain set:{}.".format(val))
                    self._gain = int(self.camera.get(cv2.CAP_PROP_GAIN))
                    self.handle_log(logging.INFO, "[OpenCV]: Gain is:{}.".format(self._gain))
                else:
                    self.handle_log(logging.ERROR, "[OpenCV]: Failed to set Gain to:{}.".format(val))
        else:
            self.handle_log(logging.CRITICAL, "[OpenCV]: Failed to set Gain, camera not open!")

    @property
    def wbtemperature(self):
        """ return opencv camera white balance temperature """
        if self.camera_open:
            return int(self.camera.get(cv2.CAP_PROP_WB_TEMPERATURE))
        else: return float("NaN")
    @wbtemperature.setter
    def wbtemperature(self, val):
        """ set opencv camera white balance temperature """
        if val is None or val < 0:
            self.handle_log(logging.WARNING, "[OpenCV]: Skipping set WB_TEMPERATURE to:{}.".format(val))
            return
        if self.camera_open:
            with self.camera_lock:
                if self.camera.set(cv2.CAP_PROP_WB_TEMPERATURE, val):
                    self.handle_log(logging.INFO, "[OpenCV]: WB_TEMPERATURE set:{}.".format(val))
                    self._wbtemp = int(self.camera.get(cv2.CAP_PROP_WB_TEMPERATURE))
                    self.handle_log(logging.INFO, "[OpenCV]: WB_TEMPERATURE is:{}.".format(self._wbtemp))
                else:
                    self.handle_log(logging.ERROR, "[OpenCV]: Failed to set whitebalance temperature to:{}.".format(val))
        else:
            self.handle_log(logging.CRITICAL, "[OpenCV]: Failed to set whitebalance temperature, camera not open!")

    @property
    def autowb(self):
        """ return opencv camera auto white balance """
        if self.camera_open:
            return int(self.camera.get(cv2.CAP_PROP_AUTO_WB))
        else: return float("NaN")
    @autowb.setter
    def autowb(self, val):
        """ set opencv camera auto white balance """
        if val is None or val < 0:
            self.handle_log(logging.WARNING, "[OpenCV]: Skipping set AUTO_WB to:{}.".format(val))
            return
        if self.camera_open:
            with self.camera_lock:
                if self.camera.set(cv2.CAP_PROP_AUTO_WB, val):
                    self.handle_log(logging.INFO, "[OpenCV]: AUTO_WB:{}.".format(val))
                    self._autowb = int(self.camera.get(cv2.CAP_PROP_AUTO_WB))
                    self.handle_log(logging.INFO, "[OpenCV]: AUTO_WB is:{}.".format(self._autowb))
                else:
                    self.handle_log(logging.ERROR, "[OpenCV]: Failed to set auto whitebalance to:{}.".format(val))
        else:
            self.handle_log(logging.CRITICAL, "[OpenCV]: Failed to set auto whitebalance, camera not open!")
