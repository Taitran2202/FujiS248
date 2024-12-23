import os
import shutil
import time
import cv2

import numpy as np
import onnxruntime as ort
from tools.ONNXRuntime import ModelState, ONNXRuntime


class TextDetection(ONNXRuntime):
    MaxDimension = 960
    MinDiv = 32
    input_width = 224
    input_height = 224
    num_channel = 3
    def __init__(self,dir):
        self.ModelDir = dir
        if not os.path.exists(self.ModelDir):
            os.mkdir(self.ModelDir)
        modelFile = os.path.join(dir, "model.onnx")
        # if not os.path.exists(modelFile):
        #     TextDetection.CopyModel(dir)

    def ImagePreprocess(self,img, targetSize):
        nw = targetSize[0]
        nh = targetSize[1]
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        resizedImg = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        # mean=[0.485, 0.456, 0.406]
        # std=[0.229, 0.224, 0.225]
        imageFloat = resizedImg / 255.0
        # normalized = (imageFloat-mean)/std
        
        return imageFloat
    
    
    def LoadOnnx(self, directory):
        try:
            providers = self.CreateProviderOption(directory)
            providers=[
    ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),
    "CPUExecutionProvider"
  ]
            options = ort.SessionOptions()
            model_path = os.path.join(directory, "simplify_model.onnx")
            model_path = os.path.join(directory, "model_rec1.onnx")
            self.ONNXSession = ort.InferenceSession(model_path, options=options,providers=providers)
            # print(self.ONNXSession.get_providers())
            self.inputs_name = [input.name for input in self.ONNXSession.get_inputs()]
            self.outputs_name = [output.name for output in self.ONNXSession.get_outputs()]
            if self.inputs_name[0] != None:
                _,self.num_channel,self.input_width,self.input_height = self.ONNXSession.get_inputs()[0].shape
            self.ONNXSession.run(self.outputs_name,
                                {
                                    self.inputs_name[0]:np.zeros((1,3,self.input_width,self.input_height),dtype=np.float32),
                                    self.inputs_name[1]:np.array([50],dtype=np.float32)
                                })
            self.State = ModelState.Loaded
        except:
            return False
        return True

    def LoadRecipe(self):
        try:
            return self.LoadOnnx(self.ModelDir)
        except:
            return False

    def Infer(self,imgInput,threshold=0.25):
        originalw=imgInput.shape[1]
        originalh=imgInput.shape[0]   
        y_scale = originalh / self.input_height 
        x_scale = originalw / self.input_width
        image_channels = imgInput.shape[-1] if imgInput.ndim == 3 else 1
        if image_channels == 1:
            imgInput = cv2.merge((imgInput,imgInput,imgInput))     
        # imgInput = cv2.cvtColor(imgInput, cv2.COLOR_BGR2RGB)           
        resized = cv2.resize(imgInput,(self.input_width,self.input_height))
        expand = np.transpose(resized,(2,0,1))
        expand = np.expand_dims(expand,0)
        normalized = expand/255.0
        normalized = np.asarray(normalized,dtype=np.float32)
        # s = time.time()
        pred_boxes = self.ONNXSession.run(self.outputs_name,
                                {
                                    self.inputs_name[0]:normalized,
                                    self.inputs_name[1]:np.array([threshold],dtype=np.float32)
                                })
        # print(time.time()-s)
        return np.array([np.array([b[0]*x_scale,b[1]*y_scale,b[2]*x_scale,b[3]*y_scale,b[4],b[5],b[6]]) for b in pred_boxes[0]]) #array shape [-1,7] x1 ,y1, x2, y2, angle, confidence, class
    
    
    
