import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import torch
from torchvision import utils
import datetime
import copy
from copy import deepcopy
from tqdm import tqdm
from time import sleep


classes = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog',
           'horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']

idx2label = dict(enumerate(iter(classes)))
label2idx = dict(zip(iter(idx2label.values()),iter(idx2label.keys()))) 
eps = 1e-10

new_size = (448,448)
no_grids = 7
no_classes = 20
no_boxes = 2



def toTensor(image):
    
    image = image.transpose((2, 0, 1))
    
    return torch.from_numpy(image)



def resizeImage(image, newSize):
    
    '''
        A function to resize an input image to a new size
        
        Args:
            image: Image to be resized of shape (height,width,channels)
            newSize: A tuple containing the new size of the image of format (new_height,new_width)
        return:
            new_image: Image resized to the new size of shape (new_heigh,new_width,channels)
    
    '''
    
    new_image = cv2.resize(image, newSize)

    return new_image




def preProcessImage(image,newSize):
    
    resizedImage = resizeImage(image,newSize)
    
    return toTensor(resizedImage)





def showImage(image):
    
    '''
        A function show an image.
        
        Args:
            image: Image to be shown of shape (height,width,channels)
    
    '''
    
    fig = plt.figure(figsize=(25., 25.))
    plt.imshow(image)
    plt.show()
    
        
def normalize(box,relH,relW):

    '''
        A function to normalize a bounding box coordinates relative to a relative width and height.
        
        Args:
            box: A tuple containing the bounding box coordinates that will be normaliezed of format (ymin,xmin,ymax,xmax)
            relH: The relative height.
            relW: The relative width.
        return:
            n_box: A list containing the normalized bounding box coordinates of format (n_ymin,n_xmin,n_ymax,n_xmax)
    
    '''
    

    ymin, xmin, ymax, xmax = box

    n_xmin = xmin / relW
    n_xmax = xmax / relW
    n_ymin = ymin / relH
    n_ymax = ymax / relH
    
    n_box = [n_ymin, n_xmin, n_ymax, n_xmax]
    
    return n_box    
        
        
def unnormalize(box,relH,relW):
    
    '''
        A function to unnormalize a bounding box coordinates relative to a relative width and height..
        
        Args:
            box: A tuple containing the bounding box coordinates that will be unnormalized of format (ymin,xmin,ymax,xmax)
            relH: The relative height.
            relW: The relative width.            
        return:
            un_box: A list containing the unnormalized bounding box coordinates of format (un_ymin,un_xmin,un_ymax,un_xmax)
    
    '''
    
    
    ymin, xmin, ymax, xmax = box

    un_xmin = int(xmin * relW)
    un_xmax = int(xmax * relW)
    un_ymin = int(ymin * relH)
    un_ymax = int(ymax * relH)
    
    un_box = [un_ymin, un_xmin, un_ymax, un_xmax]
    
    return un_box


def normalizeYOLO(box,relW,relH):
    
    '''
        A function to normalize a bounding box YOLO coordinates relative to a relative width and height..
        
        Args:
            box: A tuple containing the bounding box YOLO coordinates that will be normalized of format 
                (x,y,w,h)
            relW: The relative width used to normalize the width of the box.
            relH: The relative height used to normalize the height of the box.

        return:
            n_box: A list containing the normalized bounding box coordinates of format (n_x,n_y,n_w,n_h)
    
    '''    
    
    x,y,w,h = box
    
    n_x = x/relW
    n_y = y/relH
    n_w = w/relW
    n_h = h/relH
    
    n_box = [n_x,n_y,n_w,n_h]
    
    return n_box


def unnormalizeYOLO(box,relW,relH):
    
    '''
        A function to unnormalize a bounding box YOLO coordinates relative to a relative width and height..
        
        Args:
            box: A tuple containing the bounding box YOLO coordinates that will be unnormalized of format 
                (x,y,w,h)
            relW: The relative width used to unnormalize the width of the box.
            relH: The relative height used to unnormalize the height of the box.

        return:
            n_box: A list containing the unnormalized bounding box coordinates of format (un_x,un_y,un_w,un_h)
    
    '''  
    
    x,y,w,h = box
    
    un_x = int(x*relW)
    un_y = int(y*relH)
    un_w = int(w*relW)
    un_h = int(h*relH)
    
    un_box = [un_x,un_y,un_w,un_h]
    
    return un_box


def bbox2yolo(box):
    
    '''
        A function to convert a bounding box coordinates of format (ymin,xmin,ymax,xmax) to the 
        YOLO coordinates of format (x,y,w,h).
        
        Args:
            box: A tuple containing the bounding box coordinates that will be converted to the YOLO coordinates.
        return:
            yolo_box: A list containing the converted bounding box coordinates to YOLO ones of format (x,y,w,h)
    
    '''    
    
    ymin,xmin,ymax,xmax = box
    
    x = int((xmax+xmin)/2)
    y = int((ymax+ymin)/2)
    w = xmax - xmin
    h = ymax - ymin
    
    yolo_box = [x,y,w,h]
    
    return yolo_box
    
    

def yolo2box(ybox,maxW,maxH):
    """

        A function to convert a bounding box coordinates from the YOLO format (x,y,w,h) to (ymin,xmin,ymax,xman).

        Args:
            ybox: List containing the bounding box YOLO coordinates to be converted.
            maxW: The maximum width of the box.
            maxH: The maximum height of the box.

        return:
            box: A list containing the coordinates of the bounding box.

    """

    x, y, w, h = ybox

    xmin = max(0,(x ) - ((w ) / 2))
    ymin = max(0,(y ) - ((h ) / 2))
    xmax = max(0,min(maxW,(x ) + ((w ) / 2)))
    ymax = max(0,min(maxH,(y ) + ((h ) / 2)))

    box = [ymin, xmin, ymax, xmax]

    return box



def localizeBoxes(image, boxes):
    """
    
        A function to localize and put a group of bounding boxes on a certain image.
        
        Args:
            image: The image on which the bounding boxes will be put on it.
            boxes: A numpy array containing the boxes to be put on the image of shape 
                    (no_of_boxes, no_of_coordinates).
        
        return:
            image_R: The image with bounding boxes put on it.
               
    """

    height, width = image.shape[:2]

    image_R = np.copy(image)
    for i, box in enumerate(boxes):
        ymin, xmin, ymax, xmax = box

        xmin = max(0, int(xmin))
        xmax = min(width, int(xmax))
        ymin = max(0, int(ymin))
        ymax = min(height, int(ymax))

        start_point = (xmin, ymin)
        end_point = (xmax, ymax)
        color = (0, 0, 255)
        thickness = 1
        
        image_R = cv2.rectangle(image_R, start_point, end_point, color, thickness)

    return image_R


def localizeObjects(image, objects,aspectRatios=(1,1)):
    """
    
        A function to localize and put a group of objects labels and bounding boxes on a certain image.
        
        Args:
            image: The image on which the objects bounding boxes will be put on it.
            objects: A dictionary containing the labels and the bounding boxes coordinates of the objects.
            aspectRatios: A tuple containing width and height aspect ratios if the image has been resized.
        
        return:
            image_T: The image with objects labels and bounding boxes put on it.
               
    """

    height, width = image.shape[:2]
    h_ratio,w_ratio = aspectRatios

    image_R = np.copy(image)
    for i, obj in enumerate(objects):
        
        label = obj["name"]
        bbox = obj["bndbox"]
        conf = str(round(obj["conf"]*100,2))+"%"
        if not isinstance(bbox, dict):
            ymin, xmin, ymax, xmax = [int(b) for b in bbox]
        else:
            ymin, xmin, ymax, xmax = int(bbox["ymin"]),int(bbox["xmin"]),int(bbox["ymax"]),int(bbox["xmax"])
            ymin, xmin, ymax, xmax = int(ymin*h_ratio),int(xmin*w_ratio),int(ymax*h_ratio),int(xmax*w_ratio)

        start_point = (xmin, ymin)
        end_point = (xmax, ymax)
        color = (255, 0, 0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        orgLabel = (xmin, ymin)
        orgConf = (xmax,ymin)
        fontScale = 0.5
        textColor = (255, 0, 255)
        thickness = 1
        
        image_T = cv2.putText(image_R, label, orgLabel, font,
                              fontScale, textColor, thickness, cv2.LINE_AA)
        image_E = cv2.putText(image_T, conf, orgConf, font,
                              fontScale, textColor, thickness, cv2.LINE_AA)

        image_R = cv2.rectangle(image_E, start_point, end_point, color, thickness)

    return image_R



def iou_bb(boxA, boxB):
    """
        A function to compute in the intersection over union "IOU" between two bounding boxes.
    Args:
        boxA: The first box coordinates (ymin, xmin, ymax, xmax).
        boxB: The second box coordinates (ymin , xmin, ymax, xmax).

    Returns:
        iou: IOU value between the two bounding boxes.

    """
    eps = 1e-10
    # determine the (x, y)-coordinates of the intersection rectangle
    yA = max(boxA[0], boxB[0])
    xA = max(boxA[1], boxB[1])
    yB = min(boxA[2], boxB[2])
    xB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea+eps)
    # return the intersection over union value
    return iou



def iou_ybb(boxA,boxB):
        
    xy_A,wh_A = boxA[:2],boxA[2:]
    xy_B,wh_B = boxB[:2],boxB[2:]
    
    intersect_wh = torch.maximum(torch.zeros_like(wh_A), (wh_A + wh_B)/2 - torch.abs(xy_A - xy_B) )
    intersect_area = intersect_wh[0] * intersect_wh[1]
    
    true_area = wh_B[0] * wh_B[1]
    pred_area = wh_A[0] * wh_A[1]
    
    union_area = pred_area + true_area - intersect_area+eps
    iou = intersect_area / union_area
    
    return iou


def iou_grid(gridA,gridB):
      
    xy_A,wh_A = gridA[0],gridA[1]
    xy_B,wh_B = gridB[0],gridB[1]
            
    intersect_wh = torch.maximum(torch.zeros_like(wh_A), (wh_A + wh_B)/2 - torch.abs(xy_A - xy_B) )
    intersect_area = intersect_wh[:,:,:,:,0] * intersect_wh[:,:,:,:,1]
    
    true_area = wh_B[:,:,:,:,0] * wh_B[:,:,:,:,1]
    pred_area = wh_A[:,:,:,:,0] * wh_A[:,:,:,:,1]
    
    union_area = pred_area + true_area - intersect_area+eps
    iou = intersect_area / union_area
    
    return iou


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = \
            box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = \
            box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou





def createLabels(data):
    
    grid_size = tuple([int(s/no_grids) for s in new_size])
    widthRange = [0,new_size[1]//1]
    heightRange = [0,new_size[0]//1]
    no_samples = len(data)
    
    Y = np.zeros((no_samples,no_grids,no_grids,no_boxes,5+no_classes))
    Y_final = np.zeros((no_samples,no_grids,no_grids,(no_boxes*5)+no_classes))
    
    for i in range(no_samples):
        sample = data[i]
        
        size = sample[1]["annotation"]["size"]
        width = int(size["width"])
        height = int(size["height"])

        h_ratio = new_size[0]/height
        w_ratio = new_size[1]/width

        objects = sample[1]["annotation"]["object"]
        
        
        for o,obj in enumerate(objects):
            isLocalized = False
            labels = np.zeros(20)
            
            label = obj["name"]
            bbox = obj["bndbox"]
            
            labels[label2idx[label]]=1
            
            ymin, xmin, ymax, xmax = int(bbox["ymin"]),int(bbox["xmin"]),int(bbox["ymax"]),int(bbox["xmax"])
            ymin, xmin, ymax, xmax = int(ymin*h_ratio),int(xmin*w_ratio),int(ymax*h_ratio),int(xmax*w_ratio)
            
            r_box = [ymin,xmin,ymax,xmax]
            x, y, w, h = bbox2yolo([ymin,xmin,ymax,xmax])
            x_n, y_n, w_n, h_n = normalizeYOLO([x, y, w, h],new_size[1],new_size[0])
            
            r = int(y/grid_size[0])
            c = int(x/grid_size[1])

            b = np.argmin(Y[i,r,c,:,0])
            Y[i,r,c,b] = np.array([1,x_n,y_n,w_n,h_n,*labels])
                            
            highest = Y[i,r,c,b,5:]
            Y_final[i,r,c] = np.array([*Y[i,r,c,b,:5],*Y[i,r,c,1-b,:5],*highest])
                    
    
    return Y_final



def localizeAnnotations(image,annotations,thresh=0.5):

    h,w = image.shape[:2]
    
    for r in range(annotations.shape[0]):
        for c in range(annotations.shape[1]):
            box1 = [*annotations[r,c,:5]]
            box2 = [*annotations[r,c,5:10]]
            
            conf = box1[0]
            box = yolo2box(unnormalizeYOLO([*box1[1:5]],w,h),w,h)
            if box1[0] >= thresh:
                obj = {}
                obj["bndbox"] = box
                obj["name"] = idx2label[np.argmax(annotations[r,c,10:])]
                obj["conf"] = conf
                image = localizeObjects(image,[obj])
            else:
                pass
                #image = localizeBoxes(image,[box])
            conf = box2[0]
            box = yolo2box(unnormalizeYOLO([*box2[1:5]],w,h),w,h)
            if box2[0] >= thresh:
                obj = {}
                obj["bndbox"] = box
                obj["name"] = idx2label[np.argmax(annotations[r,c,10:])]
                obj["conf"] = conf
                image = localizeObjects(image,[obj])
            else:
                pass
                #image = localizeBoxes(image,[box])
                
    return image



def showSample(sample,thresh=0.5,isShow = True):
    
    image = sample["image"].numpy().transpose((1,2,0))
    annotations = sample["annotation"].numpy()
    
    image = localizeAnnotations(image,annotations,thresh)
    
    if isShow:
        showImage(image)
    
    return image 


    
def show_annotations_batch(sample_batched):
    
    images_batch, annotations_batch = \
            sample_batched['image'], sample_batched['annotation']
    batch_size = len(images_batch)
    
    images = []
    for i in range(batch_size):
        image = np.ascontiguousarray(images_batch[i].numpy().transpose((1,2,0)), dtype=np.uint8)
        images.append(torch.tensor(localizeAnnotations(image,annotations_batch[i].numpy()).transpose((2,0,1))))
    
    grid = utils.make_grid(images)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.title('Batch from dataloader')
    

    
def yoloLoss(labels,predictions,lambda_coords,lambda_noobj):
    
    
    N,R,C,D = labels.shape
    
    real_bbox = labels[:,:,:,:10].view((N,R,C,2,5))
    pred_bbox = predictions[:,:,:,:10].view((N,R,C,2,5))    
    
    real_classes = labels[:,:,:,10:]
    pred_classes = predictions[:,:,:,10:]
    
    real_conf = real_bbox[:,:,:,:,0]
    pred_conf = pred_bbox[:,:,:,:,0]
    
    real_xy = real_bbox[:,:,:,:,1:3]
    pred_xy = pred_bbox[:,:,:,:,1:3]
    
    real_wh = real_bbox[:,:,:,:,3:]
    pred_wh = pred_bbox[:,:,:,:,3:]

    
    conf_loss = 0
    xy_loss = 0
    wh_loss = 0
    classes_loss = 0
    
    xy_loss = torch.sum(((real_xy - pred_xy)**2)*real_conf.unsqueeze(4))*lambda_coords
    wh_loss = torch.sum(((torch.sqrt(real_wh) - torch.sqrt(pred_wh))**2)*real_conf.unsqueeze(4))*lambda_coords

    classes_loss = torch.sum(((real_classes - pred_classes)**2)*torch.max(real_conf,dim=3)[0].unsqueeze(3))
    
    obj_conf_loss = torch.sum(((real_conf-pred_conf)**2)*real_conf)
    noobj_conf_loss =  torch.sum(((real_conf-pred_conf)**2)*(1-real_conf))*lambda_noobj
    
    
    return (obj_conf_loss+noobj_conf_loss+classes_loss+xy_loss+wh_loss) 
    


def toDict(names,boxes,isGroundTruth = False,defaultDict = None):
    
    N,R,C,D = boxes.shape    
    bbox = boxes[:,:,:,:10].view((N,R*C,2,5)).reshape((N,R*C*2,5))
    classes = boxes[:,:,:,10:].view((N,R*C,20)).reshape((N,R*C,20))
    
    dic = {}
    if defaultDict is not None:
        dic = defaultDict
    
    
    for n in range(N):
        for i in range(R*C*2):
            if names[n] not in dic:
                dic[names[n]] = {"bbox":[],"scores":[],"classes":[]}
            if (isGroundTruth and bbox[n,i,0] == 1) or not isGroundTruth:
                dic[names[n]]["scores"].append(bbox[n,i,0].tolist())
                dic[names[n]]["bbox"].append(bbox[n,i,1:].tolist())
                dic[names[n]]["classes"].append(np.argmax(classes[n,i//2]))
    return dic
    

def get_model_scores(pred_boxes):
    """Creates a dictionary of from model_scores to image ids.
    Args:
        pred_boxes
    Returns:
        dict: keys are model_scores and values are image ids (usually filenames)
    """
    
    model_score={}
    for name in pred_boxes.keys():
        for i,score in enumerate(pred_boxes[name]["scores"]):
            if score not in model_score.keys():
                model_score[score] = {"name":[name],
                                      "bbox":[pred_boxes[name]["bbox"][i]],
                                      "classes":[pred_boxes[name]["classes"][i]],
                                      "tp":[pred_boxes[name]["tp"][i]]}
            else:
                model_score[score]["name"].append(name)
                model_score[score]["bbox"].append(pred_boxes[name]["bbox"][i])
                model_score[score]["classes"].append(pred_boxes[name]["classes"][i])
                model_score[score]["tp"].append(pred_boxes[name]["tp"][i])
                
    return model_score 


    
def get_batch_statistics(outputs, targets, iou_threshold = 0.5):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    names = targets.keys()
    for name in names:

        if outputs[name] is None:
            continue

        output = outputs[name]
        pred_boxes = output["bbox"]
        pred_scores = output["scores"]
        pred_labels = output["classes"]
        
        sorted_indices = np.argsort(pred_scores)[::-1]
        
        pred_scores_sorted = np.array(pred_scores)[sorted_indices].tolist()
        pred_boxes_sorted = np.array(pred_boxes)[sorted_indices].tolist()
        pred_labels_sorted = np.array(pred_labels)[sorted_indices].tolist()
        

        true_positives = np.zeros(len(pred_boxes_sorted))

        annotations = targets[name]
        target_labels = annotations["classes"]
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations["bbox"]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes_sorted, pred_labels_sorted)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                # Filter target_boxes by pred_label so that we only match against boxes of our own label
                filtered_target_position, filtered_targets = zip(*filter(lambda x: target_labels[x[0]] == pred_label, enumerate(target_boxes)))
                # Find the best matching target for our predicted box
                iou, box_filtered_index = bbox_iou(torch.tensor(pred_box).unsqueeze(0), 
                                                   torch.tensor(filtered_targets),False).max(0)

                # Remap the index in the list of filtered targets for that label to the index in the list with all targets.
                box_index = filtered_target_position[box_filtered_index]

                # Check if the iou is above the min treshold and i
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
                    
                outputs[name]["bbox"] = pred_boxes_sorted
                outputs[name]["scores"] = pred_scores_sorted
                outputs[name]["classes"] = pred_labels_sorted
                outputs[name]["tp"] = true_positives
                
    return outputs  


def ap(true,preds):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
        
    ap, p, r = [], [], []
    fpc = 0
    tpc = 0
    n_gt = 0
    n_p = 0
        
    for name in true.keys():
        n_gt += len(true[name]["bbox"])  # Number of ground truth objects
        n_p += len(preds[name]["bbox"])  # Number of predicted objects

    print(n_gt,n_p)
     
                
    model_scores = get_model_scores(preds)
    scores = np.array(list(model_scores.keys()))
    sorted_scores_indices = np.argsort(scores)[::-1]
    sorted_scores = scores[sorted_scores_indices]
        
    for score in tqdm(sorted_scores,"Computing ap"):
        
        pred_tp = model_scores[score]["tp"]

            # Create Precision-Recall curve and compute AP for each class
            
        for tp in pred_tp:

            if n_p == 0 and n_gt == 0:
                continue
            elif n_p == 0 or n_gt == 0:
                ap.append(0)
                r.append(0)
                p.append(0)
            else:
                # Accumulate FPs and TPs
                fpc += (1 - tp)
                tpc += (tp)

                # Recall
                recall_curve = tpc / (n_gt)
                r.append(recall_curve)

                # Precision
                precision_curve = tpc / (tpc+fpc+eps)
                p.append(precision_curve)

                # AP from recall-precision curve
                ap.append(compute_ap(r, p))

    # Compute F1 score (harmonic mean of precision and recall)
    plt.plot(r,p)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = (2 * p * r) / (p + r + 1e-16)

    return p, r, ap[-1]*100, f1[-1]



def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
    


    
def MAP():
    pass
    
    

def train_model(model, trainLoaders, optimizer, num_epochs=1, device = "cpu", isSave = False, filename = "mobilenet"):
    since = datetime.datetime.now()
    
    loss_train_history = []
    loss_val_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')    


    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        epoch_loss = 0
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for i,data in enumerate(tqdm(trainLoaders[phase])):

                inputs = data["image"].to(device)
                labels = data["annotation"].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):

                    # forward
                    outputs =  model(inputs/255).view(labels.shape)
                    loss = yoloLoss(labels.float(),outputs.float(),5,0.5)
                    
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()
                print(f' Iteration Loss: {loss.item()}')
                
                             
            print(f"{phase} prev epoch Loss: {epoch_loss}")
        
            epoch_loss = running_loss / len(trainLoaders[phase])

            print(f"{phase} next epoch Loss: {epoch_loss}")
            
            if phase == "val":
                
                # deep copy the model
                if epoch_loss<best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    if isSave:
                        torch.save(model.state_dict(), f"trained/{filename}")                

    print()

    time_elapsed = (datetime.datetime.now() - since).total_seconds()
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))


    # load best model weights
    model.load_state_dict(best_model_wts)
      
    return model
    

    
    
def test_model(model, test_data, device = "cpu"):
    since = datetime.datetime.now()
    
    model.eval()   # Set model to evaluate mode

    running_loss = 0.0
    ap = 0.0
    aps = 0.0
    t_dict = {}
    p_dict = {}
    
    # Iterate over data.
    for i,data in enumerate(tqdm(test_data)):
        
        names = data["name"]
        inputs = data["image"].to(device)
        labels = data["annotation"].to(device)

        # forward
        outputs =  model(inputs/255).view(labels.shape)
        loss = yoloLoss(labels.float(),outputs.float(),5,0.5)
        
        t_dict = toDict(names,labels,True,t_dict)
        p_dict = toDict(names,outputs,False,p_dict)
        
        #ap = AP(t_dict,p_dict)["ap"]
        
        # statistics
        running_loss += loss.item()
        #aps += ap
        print(f' Iteration Loss: {loss.item()}')
                             
                    
    epoch_loss = running_loss / len(test_data)
    #ap = aps / len(test_data)

    print(f" Final Loss: {epoch_loss} , Final Average Precision : {ap}")
          

    print()

    time_elapsed = (datetime.datetime.now() - since).total_seconds()
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
        
    return epoch_loss,(t_dict,p_dict)

    
    