import os

import cv2
import numpy as np
from tools.ONNXRuntime import ModelState, ONNXRuntime
from tools.utils import build_post_process
import onnxruntime as ort

class TextDetectionV2(ONNXRuntime):
    MaxDimension = 960
    MinDiv = 32
    input_width = 224
    input_height = 224
    num_channel = 3
    inspection_regions = []
    image = None
    def __init__(self,dir):
        self.ModelDir = dir
        if not os.path.exists(self.ModelDir):
            os.mkdir(self.ModelDir)
        self.thresh = 0.3
        self.box_thresh = 0.6
        self.max_candidates = 200
        self.unclip_ratio = 1.5
        self.min_size = 3
        
        postprocess_params={}
        postprocess_params['name'] = 'DBPostProcess'
        postprocess_params["thresh"] = self.thresh
        postprocess_params["box_thresh"] = self.box_thresh
        postprocess_params["max_candidates"] = self.max_candidates
        postprocess_params["unclip_ratio"] = self.unclip_ratio
        postprocess_params["use_dilation"] = False
        postprocess_params["score_mode"] = 'fast'
        self.postprocess_op = build_post_process(postprocess_params)
    def ImagePostProcess(self,results,shape_list):
        print('ImagePostProcess')
        preds = {}
        preds['maps'] = results[0]
        post_result = self.postprocess_op(preds,shape_list)
        dt_boxes = post_result[0]['points']
        
        return dt_boxes
    def ImagePreprocess(self,img, targetSize):
        print('ImagePreprocess')
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
    
       
    def CalculateResize(self,w, h, limit_side_len = 960, MinDiv = 32):
        print('CalculateResize')
        ratio = 1
        if max(h, w) > limit_side_len:
            if h>w:
                ratio = float(limit_side_len) / h
            else:
                ratio = float(limit_side_len) / w
        else:
            ratio = 1

        resize_h = int(h * ratio)
        resize_w = int(w * ratio)

        resize_h = max(int(round(float(resize_h) / MinDiv) * MinDiv), MinDiv)
        resize_w = max(int(round(float(resize_w) / MinDiv) * MinDiv), MinDiv)
        return (resize_w, resize_h)
    def LoadRecipe(self):
        print('LoadRecipe')
        try:
            return self.LoadOnnx(self.ModelDir)
        except:
            return False
    def LoadOnnx(self, directory):
        print('LoadOnnx')
        try:
            providers = self.CreateProviderOption(directory)
            options = ort.SessionOptions()
            model_path = os.path.join(directory, "model_det.onnx")
            self.ONNXSession = ort.InferenceSession(model_path, options=options,providers=providers)
            self.input_name = self.ONNXSession.get_inputs()[0].name
            self.outputs_name = [output.name for output in self.ONNXSession.get_outputs()]
            if self.input_name != None:
                _,self.num_channel,self.input_width,self.input_height = self.ONNXSession.get_inputs()[0].shape
            # if self.input_height==-1 or self.input_width==-1:
            #     self.Infer(np.zeros((1,self.num_channel,self.MinDiv,self.MinDiv),dtype=np.float32))
            # else:
            #     self.Infer(np.zeros((1,self.num_channel,self.input_height,self.input_width),dtype=np.float32))
            data = np.zeros((1,self.num_channel,224,224),dtype=np.float32)
            self.ONNXSession.run(self.outputs_name,{self.input_name:data})
            self.State = ModelState.Loaded
        except:
            return False
        return True


    def Infer(self,imgInput):
        print('Infer')
        self.image = imgInput.copy()
        if len(imgInput.shape) == 2 and self.num_channel == 3:
            imgInput = cv2.merge((imgInput,imgInput,imgInput))                
        
        if len(self.inspection_regions)!=0:
            dt_boxes = []
            for region in self.inspection_regions:
                x,y,w,h = region
                cropped_image = imgInput[y:y+h, x:x+w]
                h,w,_ = cropped_image.shape
                resize_w,resize_h = self.CalculateResize(w, h, self.MaxDimension, self.MinDiv)
                imageNormalize = self.ImagePreprocess(cropped_image,(resize_w,resize_h))
                expand = np.expand_dims(imageNormalize,0)
                expand = np.transpose(expand,(0,3,1,2))
                expand = np.asarray(expand,dtype=np.float32)
                results = self.ONNXSession.run(self.outputs_name,{self.input_name:expand})
                ratio_w = resize_w/w
                ratio_h = resize_h/h
                shape_list = [(h,w,ratio_h,ratio_w)]
                dt_box = self.ImagePostProcess(results,shape_list)
                dt_box = self.filter_tag_det_res(dt_box, cropped_image.shape)
                # dt_box = [[b[0]+x,b[1]+y,b[2],b[3],b[4]] for b in dt_box]
                dt_box = [[(x + p1, y + p2) for p1, p2 in p] for p in dt_box]
                dt_boxes.extend(dt_box)
            return dt_boxes
        else: 
            h,w,_ = imgInput.shape
            resize_w,resize_h = self.CalculateResize(w, h, self.MaxDimension, self.MinDiv)
            imageNormalize = self.ImagePreprocess(imgInput,(resize_w,resize_h))
            expand = np.expand_dims(imageNormalize,0)
            expand = np.transpose(expand,(0,3,1,2))
            expand = np.asarray(expand,dtype=np.float32)
            results = self.ONNXSession.run(self.outputs_name,{self.input_name:expand})
            ratio_w = resize_w/w
            ratio_h = resize_h/h
            shape_list = [(h,w,ratio_h,ratio_w)]
            dt_boxes = self.ImagePostProcess(results,shape_list)
            dt_boxes = self.filter_tag_det_res(dt_boxes, imgInput.shape)
            return dt_boxes
    
    def order_points_clockwise(self, pts):
        print('order_points_clockwise')
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def clip_det_res(self, points, img_height, img_width):
        print('clip_det_res')
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        print('filter_tag_det_res')
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes