import os
import time
import cv2
import colorsys
import numpy as np
import json
from components.USBIO import USBIO
from tools.ONNXRuntime import ModelState, ONNXRuntime
import onnxruntime as ort 
import concurrent.futures
from os import environ
import threading
import timeit
cv2.setUseOptimized(True)
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 4
font_color = (0, 0, 255)  # White color in BGR
font_thickness = 10
line_spacing = 30  # Adjust this to control the vertical spacing between text lines
starting_position = (10, 30)  # Adjust the (x, y) position as needed

def draw_text_with_background(image,  text, position=(10, 10), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1,font_thickness = 10, text_color=(255, 255, 255), background_color=(0, 0, 0)):

    # Create a background rectangle
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
    cv2.rectangle(image, (position[0], position[1]), (position[0] + text_width, position[1]+2*baseline+ text_height), background_color, -1)

    # Put the text on the image          
    cv2.putText(image, text, (position[0], position[1] + text_height + baseline), font, font_scale, text_color, thickness=font_thickness,lineType=cv2.LINE_AA)
    return position[1]+2*baseline+ text_height


def apply_threshold(image_draw,image, mask_threshold,mask_color,image_w,image_h):
    #print(cv2.useOptimized())
    _,binary_mask  = cv2.threshold(image,mask_threshold,255,cv2.THRESH_BINARY)
    # # start = time.time()
    binary_mask = binary_mask.astype(np.uint8)
    # print(time.time()-start)
    #binary_mask = cv2.resize(binary_mask, (image_w, image_h))
    h,w= binary_mask.shape
    h_ratio = image_h/h
    w_ratio = image_w/w
    #segmentation_indexes = np.where(segmentation_map > 0.0)
    #start = time.time()
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        contour[:, 0, 0] = contour[:, 0, 0] * w_ratio
        contour[:, 0, 1] = contour[:, 0, 1] * h_ratio
    # ret,labels,stats,centroids =cv2.connectedComponentsWithStats(image)
    # num_labels=labels.max()                      
    # for i in range(1,num_labels+1):
    #     if(stats[i, cv2.CC_STAT_AREA]>5):
    #         x=stats[i, cv2.CC_STAT_LEFT ]
    #         y=stats[i, cv2.CC_STAT_TOP ]
    #         x1=stats[i, cv2.CC_STAT_WIDTH ]+x
    #         y1=stats[i, cv2.CC_STAT_HEIGHT ]+y
    #         cv2.rectangle(image_draw,(int(x),int(y)),(int((x1)),int((y1))),(200,0,0),2)
    
    # print(time.time()-start)
    cv2.drawContours(image_draw, contours, -1, mask_color, int(image_h/400))
    
    return contours

def draw_box(image, pred_masks=None, gt_masks=None, classes_map=None, mask_threshold=0.5, colors=None):
    image_h, image_w, _ = image.shape
    num_classes = len(classes_map)
    detected_classes = []
    if colors is None:
        hsv_tuples = [(1.0 * x / num_classes, 1., 1.)
                      for x in range(num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
        
    starting_position = (10, 10)
    for i in range (num_classes):
        contours = apply_threshold(image,pred_masks[i], classes_map[i]['score_threshold'],colors[i], image_w,image_h)
        min_size = classes_map[i]['min_size']
        for contour in contours:
            area = cv2.contourArea(contour=contour)
            if(area>=min_size):
                detected_classes.append(classes_map[i]['name'])
                classes_map[i]['count']=classes_map[i]['count']+1
                next_y = draw_text_with_background(
                    image,
                    f'{classes_map[i]["name"]} : {area}',
                    starting_position,
                    font,
                    (image_h/800),
                    int(image_h/800),
                    font_color,
                    (255,255,255)
                    )
                starting_position = (starting_position[0], next_y+10)
                break         
    return image, colors,detected_classes

class InstanceSegmentation(ONNXRuntime):
    Threshold = 0.5
    min_area = 50
    def __init__(self,dir):
        self.total_count = 0
        self.pass_count = 0
        self.fail_count = 0

        self.ModelDir = dir
        if not os.path.exists(self.ModelDir):
            os.mkdir(self.ModelDir)
        self.load()
        #self.LoadRecipe()
        if ('cam1' in dir):
            self.index = 2
        else:
            self.index = 3
    def load(self):
        self.settingPath = os.path.join(self.ModelDir,"setting.json")
        if (os.path.exists(self.settingPath)):
            with open(self.settingPath, 'r') as json_file:
                self.settingDict = json.load(json_file)
        else:
            self.settingDict=None
    def load_setting_to_classes_map(self,new_classes_map):
        if (self.settingDict == None):
            return
        classes_map = self.settingDict['classes_map']
        for item in classes_map:
            selected_class = next((item2 for item2 in new_classes_map if item2["name"] == item['name']), None)
            if (selected_class is not None):
                selected_class['score_threshold'] = item['score_threshold']
                selected_class['min_size'] = item['min_size']
    def save(self):
        self.settingPath = os.path.join(self.ModelDir,"setting.json")
        self.settingDict = dict(classes_map = self.classes_map)
        json_object = json.dumps(self.settingDict, indent=4)
        # Writing to sample.json
        with open(self.settingPath, "w") as outfile:
            outfile.write(json_object)
    def LoadOnnx(self, directory):
        try:
            classes_map=[]
            class_map_path = os.path.join(directory,'classes_map.json')
            with open(class_map_path, 'r') as json_file:
                # Use json.load() to load the JSON data from the file into a Python dictionary
                classes_map = json.load(json_file)
            self.num_classes = len(classes_map)
            classes_map= [dict(**item, count=0,score_threshold=0.5,min_size=50) for item in classes_map]
            self.classes_map=classes_map
            self.load_setting_to_classes_map(self.classes_map)
            hsv_tuples = [(1.0 * x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(
                map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
            providers = self.CreateProviderOption(directory)
            providers = [
                # ('TensorrtExecutionProvider', {
                #     'device_id': 0,
                #     "trt_fp16_enable":True,
                #     "trt_engine_cache_enable":True,
                #     "trt_engine_cache_path":directory
                # }),
                ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),
                "CPUExecutionProvider"
                ]

            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            model_path = os.path.join(directory, "model.onnx")
            if not os.path.exists(model_path):
                self.State = ModelState.NotFound
                return False
            self.ONNXSession = ort.InferenceSession(model_path,options=options,providers=providers)
            print(self.ONNXSession.get_providers())
            if self.ONNXSession is not None:
                self.input_name = self.ONNXSession.get_inputs()[0].name        
                self.outputs_name = [output.name for output in self.ONNXSession.get_outputs()]
                self.input_width, self.input_height = self.ONNXSession.get_inputs()[0].shape[-2:]
                data = np.zeros((1,3,self.input_width,self.input_height), dtype=np.float32)
                self.ONNXSession.run(self.outputs_name, {self.input_name: data})
                self.State = ModelState.Loaded
                return True
            self.State = ModelState.Unloaded
            return False
        except:
            self.State = ModelState.Error
            return False

    def LoadRecipe(self):
        try:
            return self.LoadOnnx(self.ModelDir)
        except:
            return False
    def LoadModel(self,model_path):
        classes_map=[]
        directory = os.path.dirname(model_path)
        class_map_path = os.path.join(directory,'classes_map.json')
        with open(class_map_path, 'r') as json_file:
            # Use json.load() to load the JSON data from the file into a Python dictionary
            classes_map = json.load(json_file)
        self.num_classes = len(classes_map)
        self.classes_map=classes_map
        hsv_tuples = [(1.0 * x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
        try:
            providers = self.CreateProviderOption(model_path)
            providers = [
                ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),
                "CPUExecutionProvider"
                ]

            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            
            if not os.path.exists(model_path):
                self.State = ModelState.NotFound
                return False
            self.ONNXSession = ort.InferenceSession(model_path,options=options,providers=providers)
            print(self.ONNXSession.get_providers())
            if self.ONNXSession is not None:
                self.input_name = self.ONNXSession.get_inputs()[0].name
                self.outputs_name = [output.name for output in self.ONNXSession.get_outputs()]
                self.input_width, self.input_height = self.ONNXSession.get_inputs()[0].shape[-2:]
                data = np.zeros((1,3,self.input_width,self.input_height), dtype=np.float32)
                self.ONNXSession.run(self.outputs_name, {self.input_name: data})
                self.State = ModelState.Loaded
                return True
            self.State = ModelState.Unloaded
            return False
        except:
            self.State = ModelState.Error
            return False
    def Infer(self,image,thresh_value):
        if self.State != ModelState.Loaded:
            self.LoadRecipe()
        if self.State != ModelState.Loaded:
            return image
        start = time.time()
        originalw=image.shape[1]
        originalh=image.shape[0]
        image_draw = image   
        image_original = cv2.resize(image,(self.input_height,self.input_width))
        if self.ONNXSession is None:
            self.outputData['image']=np.zeros((originalh,originalw),dtype=np.uint8)
            return              
        #check channel
        channels = image_original.shape[-1] if image_original.ndim == 3 else 1
        if channels==1:
            image_original=cv2.merge((image_original,image_original,image_original))
        #im_resize = cv2.cvtColor(image_original,cv2.COLOR_BGR2RGB)
        im_resize = cv2.dnn.blobFromImage(image_original,1/255.0,size=(self.input_height,self.input_width),swapRB=True)
        image = im_resize
        # Preprocess the image (you need to adapt this based on your model's requirements)
        # For example, resizing, normalization, etc.

        # Run the model
        input_name = self.ONNXSession.get_inputs()[0].name
        #input_data = np.expand_dims(image.astype('float32')/255.0, axis=0)
        input_data=image
        thresholds = np.ones(self.num_classes, dtype=np.float32)*0.5
        output_name = self.ONNXSession.get_outputs()[0].name
        segmentation_result = self.ONNXSession.run([output_name], {input_name: input_data})[0][0]
        image_predict,_,detected_classes = draw_box(image_draw,
                                segmentation_result, None, self.classes_map, thresh_value, self.colors)
        self.total_count = self.total_count+1
        if (len(detected_classes)>0):
            self.fail_count = self.fail_count+1
            USBIO.WritePulse(1,self.index-2,1)          
        else:
            self.pass_count = self.pass_count+1           
            USBIO.WritePulse(0,self.index,1)
        print("segment_v2: {}ms".format(round((time.time() - start)*1000,2)))
        return image_predict,detected_classes