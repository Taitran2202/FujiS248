import atexit
import os
from queue import Queue
import threading
from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
from components.init_database import DbContext
from datetime import datetime
import json



class Task(object):
    def __init__(self,id_number,defect_type,image,captured_date,save_image_path=None):
        self.id_number = id_number
        self.image = image
        self.defect_type = defect_type
        self.captured_date = captured_date
        self.save_image_path = save_image_path
        


class Recorder(QtCore.QObject):
    def __init__(self,db_name, save_dir,image_process, parent=None):
        super(Recorder, self).__init__(parent)
        self.data_dir = image_process.data_dir
        self.db_name = db_name
        self.save_dir = save_dir
        self.image_process = image_process
        self.image_queue = Queue(5)
        self.folder_name_format = "%Y_%m_%d"
        self.image_name_format = "%Y_%m_%d_%H_%M_%S_%f"
        self.image_format = '.png'
        if not os.path.exists(save_dir):
            os.makedirs(self.save_dir)
        self.enable_record = False
        self.shutdown_recorder = threading.Event()
        self.shutdown_recorder.clear()
        self.triggerRecord = threading.Event()
        self.triggerRecord.clear()
        self.record_thread = threading.Thread(target=self.save_image)
        self.record_thread.start()
        self.image_process.recordReady.connect(self.record)
        self.save_folder = None
        self.load()
    def load(self):
        self.settingPath = os.path.join(self.data_dir,"record_setting.json")
        if (os.path.exists(self.settingPath)):
            with open(self.settingPath, 'r') as json_file:
                self.settingDict = json.load(json_file)
                self.save_dir = self.settingDict['save_dir']
                self.folder_name_format = self.settingDict['folder_name_format']
                self.image_name_format = self.settingDict['image_name_format']
                self.image_format = self.settingDict['image_format']
                self.enable_record = self.settingDict['enable_record']
        else:
            self.settingDict=None
    def save(self):
        self.settingPath = os.path.join(self.data_dir,"record_setting.json")
        self.settingDict = dict(save_dir = self.save_dir,
                                folder_name_format = self.folder_name_format,
                                image_name_format = self.image_name_format,
                                image_format = self.image_format,
                                enable_record = self.enable_record)
        json_object = json.dumps(self.settingDict, indent=4)
        # Writing to sample.json
        with open(self.settingPath, "w") as outfile:
            outfile.write(json_object)
    def dispose(self):
        self.shutdown_recorder.set()
        self.triggerRecord.set()
        
    def save_image(self):
        while not self.shutdown_recorder.is_set():
            while not self.image_queue.empty():
                task = self.image_queue.get()
                if task.save_image_path != None:
                    cv2.imwrite(task.save_image_path,task.image)
            self.triggerRecord.wait()
            self.triggerRecord.clear()
        print('Clean up Recoder')
    def record(self,task:Task):
        with DbContext(self.db_name) as db:
            sql = ''' INSERT INTO tasks(id_number,defect_type,image_path, captured_date)
              VALUES(?,?,?,?) '''
            image_path = None
            if self.enable_record:
                save_folder = os.path.join(self.save_dir,task.captured_date.strftime(self.folder_name_format))
                if (save_folder!= self.save_folder):
                    print('save folder changed')
                    if not os.path.exists(save_folder):
                        os.mkdir(save_folder)
                    self.save_folder = save_folder
                image_path = os.path.join(save_folder,task.captured_date.strftime(self.image_name_format)+self.image_format)
                task.save_image_path = image_path
            new_task = (task.id_number,task.defect_type,image_path,task.captured_date)
            db.execute(sql, new_task)

        if self.enable_record:
            self.triggerRecord.set()
            self.image_queue.put(task)
            # cv2.imwrite(os.path.join(self.save_dir,task.captured_date.strftime("%d_%m_%Y_%H_%M_%S")+'.png'),task.image)