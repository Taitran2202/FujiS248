import math
import os
import onnxruntime as ort

import cv2
import numpy as np
from tools.ONNXRuntime import ModelState, ONNXRuntime
from tools.utils import build_post_process


class TextRecognition(ONNXRuntime):
    MaxWidth = 500
    MinDiv = 32
    input_width = 224
    input_height = 224
    num_channel = 3
    def __init__(self,dir):
        self.ModelDir = dir
        if not os.path.exists(self.ModelDir):
            os.mkdir(self.ModelDir)
        dictfile = os.path.join(dir, "ch_dict_file.txt")
        modelFile = os.path.join(dir, "ch_PP-OCRv3_rec_infer.onnx")
        # if not os.path.exists(dictfile) or not os.path.exists(modelFile):
        #     TextRecognition.CopyModel(dir)
        if(os.path.exists(os.path.join(dir,'ch_dict_file.txt'))):
            postprocess_params = {
                'name': 'CTCLabelDecode',
                "character_dict_path": os.path.join(dir,'ch_dict_file.txt'),
                "use_space_char": True
            }
        else:
            postprocess_params = {
                'name': 'CTCLabelDecode',
                "character_dict_path": None,
                "use_space_char": True
            }
        self.postprocess_op = build_post_process(postprocess_params)
    def ImagePreprocess(self, img, targetSize):
        nw = targetSize[0]
        nh = targetSize[1]
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        resizedImg = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        # imageFloat = resizedImg.ConvertImageType("float")
        #mean std normalize 
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        imageFloat = resizedImg / 255.0
        normalized = (imageFloat-mean)/std
        
        return normalized
    
    def LoadOnnx(self, directory):
        try:
            
            providers = self.CreateProviderOption(directory)
            options = ort.SessionOptions()
            model_path = os.path.join(directory, "model_rec.onnx")
            # model_path = os.path.join(directory, "model_rec.onnx")
            self.ONNXSession = ort.InferenceSession(model_path, options=options,providers=providers)
            self.input_name = self.ONNXSession.get_inputs()[0].name
            self.outputs_name = [output.name for output in self.ONNXSession.get_outputs()]
            if self.input_name != None:
                _,self.num_channel,self.input_height,self.input_width = self.ONNXSession.get_inputs()[0].shape
            if str(self.input_height).__contains__('DynamicDimension'):
                self.input_height = 48
            if str(self.input_width).__contains__('DynamicDimension'):
                self.input_width = self.MaxWidth
            # if self.input_height==-1 or self.input_width==-1:
            #     self.Infer(np.zeros((1,self.num_channel,self.MinDiv,self.MinDiv),dtype=np.float32))
            # else:
            #     self.Infer(np.zeros((1,self.num_channel,self.input_height,self.input_width),dtype=np.float32))
            # self.InferBatch([np.zeros((self.MinDiv,self.MinDiv,self.num_channel),dtype=np.float32)],self.MaxWidth)
            self.ONNXSession.run(self.outputs_name,{self.input_name:np.zeros((1,3,48,500),dtype=np.float32)})
            self.State = ModelState.Loaded
        except:
            return False
        return True

    def LoadRecipe(self):
        try:
            if(os.path.exists(os.path.join(self.ModelDir,'ch_dict_file.txt'))):
                postprocess_params = {
                    'name': 'CTCLabelDecode',
                    "character_dict_path": os.path.join(self.ModelDir,'ch_dict_file.txt'),
                    "use_space_char": True
                }
                self.postprocess_op = build_post_process(postprocess_params)
            return self.LoadOnnx(self.ModelDir)
        except:
            return False

    
    def FindOptimalResize(self, img, MaxWidth):
        h,w = img.shape[:2]
        max_wh_ratio = self.input_width * 1.0 / self.input_height
        wh_ratio = w / float(h)
        max_wh_ratio = max(max_wh_ratio, wh_ratio)
        # dstWidth = int(self.input_height * max_wh_ratio)
        dstWidth = MaxWidth
        if math.ceil(self.input_height * wh_ratio) > dstWidth:
            resizeWidth = dstWidth
        else:
            resizeWidth = int(math.ceil(self.input_height * wh_ratio))
        return dstWidth,resizeWidth
    

    def ResizeNormalize(self,image, resizeW, resizeH, dstW, dstH):
        resizedImg = cv2.resize(image,(resizeW,resizeH),cv2.INTER_LINEAR)
        # imageFloat = resizedImg.ConvertImageType("float");
        imageFloat = padding_image(resizedImg,dstW,dstH)
        imageFloat = imageFloat / 255.0
        imageFloat -= 0.5
        imageFloat /= 0.5
        
        return imageFloat
    def InferBatch(self,imgInput, MaxWidth):
        SelectedWidth = MaxWidth
        if len(imgInput) == 0:
            return []
        result = []
        resizeWidths = []
        dstWidths = []
        # img_num = len(imgInput)
        # indices = np.argsort(np.array(width_list))
        # rec_res = [['', 0.0]] * img_num
        if self.input_width == -1:
            for i in range(len(imgInput)):
                img = imgInput[i]
                if len(img.shape) == 2 and self.num_channel == 3:
                    imgInput[i] = cv2.merge((img,img,img))
                dstWidth, resizeWidth = self.FindOptimalResize(img,SelectedWidth)
                resizeWidths.append(resizeWidth)
                dstWidths.append(dstWidth)
        else:
            for i in range(len(imgInput)):
                img = imgInput[i]
                if len(img.shape) == 2 and self.num_channel == 3:
                    imgInput[i] = cv2.merge((img,img,img)) 
                dstWidth, resizeWidth = self.FindOptimalResize(img,SelectedWidth)
                dstWidths.append(dstWidth)
                resizeWidths.append(resizeWidth)
        maxWidth = max(dstWidths)
        for i in range(len(imgInput)):
            try:
                img = imgInput[i]
                # print(img.shape)
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                imageNormalize = self.ResizeNormalize(img,resizeWidths[i],self.input_height, maxWidth, self.input_height)
                expand = np.expand_dims(imageNormalize,0)
                expand = np.transpose(expand,(0,3,1,2))
                expand = np.asarray(expand,dtype=np.float32)
                text = self.ONNXSession.run(self.outputs_name,{self.input_name:expand})
                preds = text[0]
                rec_result = self.postprocess_op(preds)
                result.append(rec_result[0][0])
            except:
                result.append('')
        return result


def padding_image(image, desired_width, desired_height):
    # Get the original image dimensions
    h, w = image.shape[:2]
    padded = np.zeros((desired_height,desired_width,3),dtype=np.float32)
    if(w>desired_width):
        padded=image[0:desired_height,0:desired_width]
    else:
        padded[0:h,0:w] = image    
    return padded