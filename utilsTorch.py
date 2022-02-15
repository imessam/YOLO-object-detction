import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import torch
from torchvision import utils


classes = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog',
           'horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']

idx2label = dict(enumerate(iter(classes)))
label2idx = dict(zip(iter(idx2label.values()),iter(idx2label.keys()))) 



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
        if not isinstance(bbox, dict):
            ymin, xmin, ymax, xmax = [int(b) for b in bbox]
        else:
            ymin, xmin, ymax, xmax = int(bbox["ymin"]),int(bbox["xmin"]),int(bbox["ymax"]),int(bbox["xmax"])
            ymin, xmin, ymax, xmax = int(ymin*h_ratio),int(xmin*w_ratio),int(ymax*h_ratio),int(xmax*w_ratio)

        start_point = (xmin, ymin)
        end_point = (xmax, ymax)
        color = (255, 0, 0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (xmin, ymin)
        fontScale = 1
        textColor = (255, 0, 255)
        thickness = 2
        image_T = cv2.putText(image_R, label, org, font,
                              fontScale, textColor, thickness, cv2.LINE_AA)

        image_R = cv2.rectangle(image_T, start_point, end_point, color, thickness)

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
    
    eps = 1e-10
    
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
    
    eps = 1e-10
    
    xy_A,wh_A = gridA[0],gridA[1]
    xy_B,wh_B = gridB[0],gridB[1]
            
    intersect_wh = torch.maximum(torch.zeros_like(wh_A), (wh_A + wh_B)/2 - torch.abs(xy_A - xy_B) )
    intersect_area = intersect_wh[:,:,:,:,0] * intersect_wh[:,:,:,:,1]
    
    true_area = wh_B[:,:,:,:,0] * wh_B[:,:,:,:,1]
    pred_area = wh_A[:,:,:,:,0] * wh_A[:,:,:,:,1]
    
    union_area = pred_area + true_area - intersect_area+eps
    iou = intersect_area / union_area
    
    return iou





def createLabelsFaster(data,new_size,no_grids,no_boxes,no_classes):
    
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



def localizeAnnotations(image,annotations):

    h,w = image.shape[:2]
    
    for r in range(annotations.shape[0]):
        for c in range(annotations.shape[1]):
            box1 = [*annotations[r,c,:5]]
            box2 = [*annotations[r,c,5:10]]
            
            box = yolo2box(unnormalizeYOLO([*box1[1:5]],w,h),w,h)
            if box1[0] >=0.8:
                obj = {}
                obj["bndbox"] = box
                obj["name"] = idx2label[np.argmax(annotations[r,c,10:])]
                image = localizeObjects(image,[obj])
            else:
                pass
                #image = localizeBoxes(image,[box])
            box = yolo2box(unnormalizeYOLO([*box2[1:5]],w,h),w,h)
            if box2[0] >=0.8:
                obj = {}
                obj["bndbox"] = box
                obj["name"] = idx2label[np.argmax(annotations[r,c,10:])]
                image = localizeObjects(image,[obj])
            else:
                pass
                #image = localizeBoxes(image,[box])
                
    return image



def showSample(sample):
    
    image = sample["image"].numpy().transpose((1,2,0))
    annotations = sample["annotation"].numpy()
    
    image = localizeAnnotations(image,annotations)
    
    showImage(image)


    
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
    

    
    
    
def accuracy(true,preds,iou_thresh = 0.5, conf_thresh = 0.5 ):
    
    N,R,C = true.shape[:3]
    
    t_classes = true[:,:,:,10:]
    p_classes = preds[:,:,:,10:]
    
    t_boxes = true[:,:,:,:10].view((N,R,C,2,5))
    p_boxes = preds[:,:,:,:10].view((N,R,C,2,5))
    
    TP = 0
    FP = 0 
    TN = 0
    FN = 0
    accuracy = 0
    
    
    for n in range(N):
        for r in range(R):
            for c in range(C):
                for b in range(2):
                    t_conf = t_boxes[n,r,c,b,0]
                    p_conf = p_boxes[n,r,c,b,0]
                    
                    t_box = t_boxes[n,r,c,b,1:]
                    p_box = p_boxes[n,r,c,b,1:]

                    if t_conf == 0:
                        if p_conf<conf_thresh:
                            TN += 1
                        else:
                            FP += 1
                    else :
                        iou = iou_ybb(p_box,t_box)
                        
                        if p_conf<conf_thresh:
                            FN += 1
                        else:
                            if iou<iou_thresh:
                                FP += 1
                            else:
                                if torch.argmax(t_classes) != torch.argmax(p_classes):
                                    FP += 1
                                else:
                                    TP += 1
    accuracy = (TP+TN)/(TP+FP+TN+FN) 
    
    return accuracy,TP,FP,TN,FN

    
    
    
def precision(true,preds,thresh):
    
    pass


def recall(true,preds,thresh):
    
    pass


def AP(precision,recall):
    
    pass


def mAP(true,preds,thresh):
    
    pass
    
    
    