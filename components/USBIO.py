from Automation.BDaq import *
from Automation.BDaq.InstantDoCtrl import InstantDoCtrl
from Automation.BDaq.InstantDiCtrl import InstantDiCtrl
from Automation.BDaq.BDaqApi import AdxEnumToString, BioFailed
from PyQt5 import QtCore, QtGui, QtWidgets
import time
import threading

class USBIO(QtCore.QObject):
    is_available = False
    __instance = None
    instantDoCtrl = None
    instantDiCtrl = None
    frameReady1 = QtCore.pyqtSignal(bool)
    frameReady2 = QtCore.pyqtSignal(bool)
    scanThread = None
    is_scanning = False
    @staticmethod 
    def instance():
        """ Static access method. """
        if USBIO.__instance == None:
            USBIO()
        return USBIO.__instance

    def __init__(self):
        """ Virtually private constructor. """
        if USBIO.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            USBIO.__instance = self
            try:
                USBIO.instantDoCtrl = InstantDoCtrl(0)
                USBIO.instantDiCtrl = InstantDiCtrl(0)
                USBIO.is_available = True
            except:
                pass
            super(USBIO,self).__init__()

            
    @staticmethod 
    def StartScan():
        if(USBIO.is_available):
            USBIO.scanThread = threading.Thread(target=USBIO.ScanIO)
            USBIO.scanThread.start()
    def StopScan():
        USBIO.is_scanning = False
        USBIO.scanThread.join()
    def ScanIO():
        USBIO.is_scanning = True   
        ret, values = USBIO.instantDiCtrl.readAny(0,1)
        bit0_pre = values[0] & 1
        bit1_pre = (values[0]>>1) & 1
        while USBIO.is_scanning:
            ret, values = USBIO.instantDiCtrl.readAny(0,1)
            bit0 = values[0] & 1
            bit1 = (values[0]>>1) & 1
            if ((bit0!=bit0_pre) & (bit0==1)):
                USBIO.__instance.frameReady1.emit(True)
                print(f'cam 1 triggered')
            if ((bit1!=bit1_pre) & (bit1==1)):
                USBIO.__instance.frameReady2.emit(True)
                print(f'cam 2 triggered')
            bit0_pre = bit0
            bit1_pre = bit1
            time.sleep(0.02)
            
        print('scan exit')
    @staticmethod 
    def WriteBit(port:int,bit:int,data:c_uint8):
        if USBIO.is_available:
            ret = USBIO.instantDoCtrl.writeBit(port,bit,data)
            print(ret)
    @staticmethod 
    def WritePulse(port:int,bit:int,data:c_uint8,pulse_width = 0.1):
        if USBIO.is_available:
            pusleThread = threading.Thread(target=USBIO._WritePulse,args=(port,bit,data,pulse_width))
            pusleThread.start()
    @staticmethod 
    def _WritePulse(port:int,bit:int,data:c_uint8,pulse_width = 0.1):
        ret = USBIO.instantDoCtrl.writeBit(port,bit,1)
        time.sleep(pulse_width)
        ret = USBIO.instantDoCtrl.writeBit(port,bit,0)
    @staticmethod 
    def ReadBit(port:int,bit:int):
        if USBIO.is_available:
            ret,value = USBIO.instantDoCtrl.readBit(port,bit)
            print(ret)
            return value
        else:
            return 0