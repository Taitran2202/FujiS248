import math
import os
import re
import sys
import threading
import time
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from pyparsing import Regex
from components.Recorder import Task
from components.USBIO import USBIO
from components.init_database import DbContext
from datetime import datetime

from tools.TextDetection import TextDetection
from tools.InstanceSegmentation import InstanceSegmentation
#from tools.TextRecognition import TextRecognition
from tools.utils.utils import CropImage, box2center, center2box, convert_format




DefectMap1=[{'id': 0, 'name': 'bot-bac', 'supercategory': 'none'},
{'id': 1, 'name': 'dom-den', 'supercategory': 'none'},
{'id': 2, 'name': 'bot-bac-dinh-keo', 'supercategory': 'none'},
{'id': 3, 'name': 'vang-keo', 'supercategory': 'none'},
{'id': 4, 'name': 'nut', 'supercategory': 'none'},
{'id': 5, 'name': 'keo-arona', 'supercategory': 'none'},
{'id': 6, 'name': 'loi-day', 'supercategory': 'none'},
{'id': 7, 'name': 'xuoc', 'supercategory': 'none'}]

class Capture(QtCore.QObject):
    started = QtCore.pyqtSignal()
    frameReady = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super(Capture, self).__init__(parent)
        self.m_frame = None
        self.m_timer = QtCore.QBasicTimer()
        self.m_videoCapture = cv2.VideoCapture()

    @QtCore.pyqtSlot()
    def start(self, cam=0):
        if self.m_videoCapture is not None:
            self.m_videoCapture.release()
            self.m_videoCapture = cv2.VideoCapture(cam)
        if self.m_videoCapture.isOpened():
            self.m_timer.start(0, self)
            self.started.emit()
    @QtCore.pyqtSlot()
    def trigger(self, cam=0):
        return
    @QtCore.pyqtSlot()
    def release(self, cam=0):
        return
    @QtCore.pyqtSlot()
    def open_image(self):
        dialog = QtWidgets.QFileDialog()
        file_name = dialog.getOpenFileName(None, "Select Image")
        if file_name:
            image = cv2.imread(file_name)
            self.frameReady.emit(image)
    @QtCore.pyqtSlot()
    def stop(self):
        self.m_timer.stop()

    def __del__(self):
        self.m_videoCapture.release()

    def frame(self):
        return self.m_frame

    def timerEvent(self, event):
        if event.timerId() != self.m_timer.timerId():
            return

        ret, val = self.m_videoCapture.read()
        if not ret:
            self.m_timer.stop()
            return
        self.m_frame = val    
        self.frameReady.emit(self.m_frame)

    frame = QtCore.pyqtProperty(np.ndarray, fget=frame, notify=frameReady, user=True)

# from vimba import *
# class OldHandler:
#     def __init__(self,queue):
#         self.shutdown_event = threading.Event()
#         self.queue=queue

#     def __call__(self, cam: Camera, frame: Frame):
#         if frame.get_status() == FrameStatus.Complete:
#             print('{} acquired {}'.format(cam, frame), flush=True)
#             msg = 'Stream from \'{}\'. Press <Q> to stop stream.'
#             try:
#                 image = frame.as_opencv_image()
#             except:
#                 image = np.zeros((frame.get_height(),frame.get_width(),3),dtype=np.uint8)
#             if(self.queue.qsize()>=10):
#                 self.queue.get()
#             self.queue.put(image)
#         cam.queue_frame(frame)
from vmbpy import *
opencv_display_format = PixelFormat.Mono8
class Handler:
    def __init__(self,queue):
        self.shutdown_event = threading.Event()
        self.queue=queue
        self.image_capture_event = threading.Event()
        self.image_capture_event.clear()

    def __call__(self, cam: Camera, stream: Stream, frame: Frame):
        print('frame recieved')
        if frame.get_status() == FrameStatus.Complete:
            print('{} acquired {}'.format(cam, frame), flush=True)
            # Convert frame if it is not already the correct format
            if frame.get_pixel_format() == opencv_display_format:
                display = frame
            else:
                # This creates a copy of the frame. The original `frame` object can be requeued
                # safely while `display` is used
                display = frame.convert_pixel_format(opencv_display_format)

            msg = 'Stream from \'{}\'. Press <Enter> to stop stream.'
            # cv2.imshow(msg.format(cam.get_name()), display.as_opencv_image())
            try:
                image = display.as_opencv_image()
            except:
                image = np.zeros((frame.get_height(),frame.get_width(),3),dtype=np.uint8)
            if(self.queue.qsize()>=10):
                self.queue.get()
            self.queue.put(image)
            self.image_capture_event.set()
        cam.queue_frame(frame)
from queue import Queue
def abort(reason: str, return_code: int = 1, usage: bool = False):
    print(reason + '\n')

    sys.exit(return_code)

def get_camera(camera_id=None) -> Camera:
    with VmbSystem.get_instance() as vmb:
        if camera_id is not None:
            if isinstance(camera_id,int):
                try:
                    cams = vmb.get_all_cameras()
                    return cams[camera_id]
                except:
                    return None
            else:
                try:
                    return vmb.get_camera_by_id(camera_id)

                except VmbCameraError:
                    return None
                # abort('Failed to access Camera \'{}\'. Abort.'.format(camera_id))

        else:
            cams = vmb.get_all_cameras()
            if not cams:
                # abort('No Cameras accessible. Abort.')
                return None
            return cams[0]
def setup_camera(cam: Camera):
    
    # # Enable auto exposure time setting if camera supports it
    # try:
    #     cam.ExposureAuto.set('Continuous')

    # except (AttributeError, VmbFeatureError):
    #     pass

    # # Enable white balancing if camera supports it
    # try:
    #     cam.BalanceWhiteAuto.set('Continuous')

    # except (AttributeError, VmbFeatureError):
    #     pass

    # Try to adjust GeV packet size. This Feature is only available for GigE - Cameras.
    # try:
        
    #     stream = cam.get_streams()[0]
    #     stream.GVSPAdjustPacketSize.run()
    #     while not stream.GVSPAdjustPacketSize.is_done():
    #         pass

    # except (AttributeError, VmbFeatureError):
    pass

def setup_pixel_format(cam: Camera):
    # Query available pixel formats. Prefer color formats over monochrome formats
    cam_formats = cam.get_pixel_formats()
    cam_color_formats = intersect_pixel_formats(cam_formats, COLOR_PIXEL_FORMATS)
    convertible_color_formats = tuple(f for f in cam_color_formats
                                      if opencv_display_format in f.get_convertible_formats())
    cam_mono_formats = intersect_pixel_formats(cam_formats, MONO_PIXEL_FORMATS)
    convertible_mono_formats = tuple(f for f in cam_mono_formats
                                     if opencv_display_format in f.get_convertible_formats())
    cam.set_pixel_format(convertible_mono_formats[0])
    return
    # if OpenCV compatible color format is supported directly, use that
    if opencv_display_format in cam_formats:
        cam.set_pixel_format(opencv_display_format)

    # else if existing color format can be converted to OpenCV format do that
    elif convertible_color_formats:
        cam.set_pixel_format(convertible_color_formats[0])

    # fall back to a mono format that can be converted
    elif convertible_mono_formats:
        cam.set_pixel_format(convertible_mono_formats[0])

    else:
        abort('Camera does not support an OpenCV compatible format. Abort.')


class GigEVisionCamera(QtCore.QObject):
    started = QtCore.pyqtSignal()
    frameReady = QtCore.pyqtSignal(np.ndarray)
    vimba_lock  = threading.Lock()
    def __init__(self,cam_id = None,setting_file=None, parent=None):
        super(GigEVisionCamera, self).__init__(parent)
        self.setting_file = setting_file
        self.m_frame = None
        self.ImageQueue = Queue()
        self.handler = Handler(self.ImageQueue)
        self.is_run = threading.Event()
        self.is_run.clear()
        self.is_exit = False
        self.is_running = True
        self.is_connected = False
        self.cam_id = cam_id
        self.InitializeCamera()
        if (self.cam is not None):
            self.thread = threading.Thread(target=self.main_loop)
            self.thread.start()
        else:
            self.thread = None
        
    @QtCore.pyqtSlot()        
    def camera_triggered(self):
        self.trigger()
        print(f'camera capture 2')
    def InitializeCamera(self):       
        self.cam = get_camera(self.cam_id)
        if (self.cam is None):
            return
        cam = self.cam
        if cam is None:
            self.is_connected=False
            return
        with cam:
            try:
                self.load_cam_setting()
                setup_pixel_format(cam)
            except Exception as error:
                print(error)
            self.is_connected=True
            self.started.emit()
    def load_cam_setting(self):
        if(os.path.exists(self.setting_file)):
            print(f'load camera setting from {self.setting_file}')
            self.cam.load_settings(self.setting_file)                
                
    def main_loop(self):
        with self.cam as cam:
            while self.is_run.wait() and self.is_running: 
                if (self.is_exit):
                    return
                self.is_run.clear()           
                try:
                    cam.get_feature_by_name('TriggerSource').set("Software")
                    cam.get_feature_by_name('AcquisitionMode').set("Continuous")
                    cam.get_feature_by_name('TriggerSelector').set("FrameStart")
                    cam.get_feature_by_name('TriggerMode').set("On")
                except Exception as error:
                    print(error)
                self.cam.start_streaming(handler = self.handler, buffer_count = 10)
                # self.handler.shutdown_event.wait()
                self.handler.shutdown_event.clear()
                while not self.handler.shutdown_event.is_set():
                    while not self.ImageQueue.empty():
                        self.m_frame = self.ImageQueue.get()
                        self.frameReady.emit(self.m_frame)
                    self.handler.image_capture_event.wait()
                    self.handler.image_capture_event.clear()
                try:
                    if (self.is_exit):
                        return
                    self.cam.stop_streaming()
                except:
                    pass
                
        
    @QtCore.pyqtSlot()
    def start(self, cam=0):
        self.is_running=True
        self.is_run.set()

    @QtCore.pyqtSlot()
    def stop(self):
        self.handler.shutdown_event.set()
        self.is_run.clear()
        self.handler.image_capture_event.set()
    @QtCore.pyqtSlot()
    def release(self):
        self.is_exit= True
        self.is_running = False
        self.is_run.set()
        if (self.thread):
            self.thread.join()
    @QtCore.pyqtSlot()
    def trigger(self):
        with self.cam as cam:
            trigger_cmd = cam.get_feature_by_name("TriggerSoftware")
            trigger_cmd.run()
    @QtCore.pyqtSlot()
    def open_image(self):
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(None, 'Select image', '', 'Images (*.png *.bmp *.jpg)')
        if file_name:
            image = cv2.imread(file_name)
            print("1")
            self.frameReady.emit(image)
    def __del__(self):
        # self.m_videoCapture.release()
        return

    def frame(self):
        return self.m_frame

    frame = QtCore.pyqtProperty(np.ndarray, fget=frame, notify=frameReady, user=True)

class ImageProcess(QtCore.QObject):
    imageReady = QtCore.pyqtSignal(QtGui.QImage)
    recordReady = QtCore.pyqtSignal(Task)
    showFPS = QtCore.pyqtSignal(float)
    def __init__(self,capture:GigEVisionCamera,data_dir="data/cam1", parent=None):
        super(ImageProcess, self).__init__(parent)
        self.data_dir = data_dir
        self.capture = capture
        
        self.m_frame = np.array([])
        self.m_image = QtGui.QImage()
        self.regex_code = "\d{2}\w{8}"
        # ModelDetectionDir = r"C:\Users\newocean\.novision\models\pretrained_small_model"
        # self.Detection = TextDetection(ModelDetectionDir)        
        ModelDetectionDir = data_dir
        self.Detection = InstanceSegmentation(ModelDetectionDir)
        #self.Detection.LoadRecipe()
        self.Detection.Minsize = 0
        #ModelRecogintionDir = r"C:\Users\NewOcean\.novision\models"
        #self.Recognition = TextRecognition(ModelRecogintionDir)
        #self.Recognition.LoadRecipe()
        self.capture.frameReady.connect(self.processFrame)
    
    def queue(self, frame):
        self.m_frame = frame
        # if not self.m_timer.isActive():
        #     self.m_timer.start(0, self)

    def process(self, frame):
        
        #image = frame.copy()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # image = CropImage(image,np.array((700,0,700,1024,0)))
        bypass=False
        if not bypass:
            st1=time.time()
            image,detection_result = self.Detection.Infer(image,self.Detection.Threshold)
            if len(detection_result)>0:
                task = Task('NG',';'.join(detection_result),frame,datetime.now())
            else:
                task = Task('OK',';'.join(detection_result),frame,datetime.now())
            #print(thresh.shape[:2])
            # for classes_map in detection_result:
            #     print(classes_map)
        else:
            task = Task('ByPass',[],frame,datetime.now())
        processed_image = image
        channels = processed_image.shape[-1] if processed_image.ndim == 3 else 1
        height, width = processed_image.shape[:2]
        image_format = QtGui.QImage.Format_Indexed8 if channels == 1 else QtGui.QImage.Format_RGB888
        bytes_per_line = channels * width
        self.m_image = QtGui.QImage(processed_image.data, width, height, bytes_per_line, image_format)
        self.imageReady.emit(QtGui.QImage(self.m_image))
        self.recordReady.emit(task)
        return

    def processFrame(self, frame):
        # if self.m_processAll:
        #     self.process(frame)
        # else:
        #     self.queue(frame)
        self.process(frame)

    def image(self):
        return self.m_image

    def display_text(self,image,text):
        t_size = cv2.getTextSize(text, 0, self.font_scale, thickness=self.bbox_thick)[0]
        # p1 = (int(boxes[i][0]-boxes[i][2]/2),int(boxes[i][1]-boxes[i][3]/2))
        p1=(50,50)
        # p2 = (int(boxes[i][0]-boxes[i][2]/2+ t_size[0]),int(boxes[i][1]-boxes[i][3]/2- t_size[1] - 3))
        p2 = (int(50 + t_size[0]),int(50 - t_size[1] - 4))
        # print(p1,p2)
        cv2.rectangle(image, p1, p2, self.bbox_color, cv2.FILLED)  # filled
        cv2.putText(image, text, (p1[0],p1[1]-2), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (255,255,255), self.bbox_thick, lineType=cv2.LINE_AA)

    image = QtCore.pyqtProperty(QtGui.QImage, fget=image, notify=imageReady, user=True)





