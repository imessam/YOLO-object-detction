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
            
            box = yolo2box(unnormalizeYOLO([*box1[1:5]],w,h),w,h)
            if box1[0] >= thresh:
                obj = {}
                obj["bndbox"] = box
                obj["name"] = idx2label[np.argmax(annotations[r,c,10:])]
                image = localizeObjects(image,[obj])
            else:
                pass
                #image = localizeBoxes(image,[box])
            box = yolo2box(unnormalizeYOLO([*box2[1:5]],w,h),w,h)
            if box2[0] >= thresh:
                obj = {}
                obj["bndbox"] = box
                obj["name"] = idx2label[np.argmax(annotations[r,c,10:])]
                image = localizeObjects(image,[obj])
            else:
                pass
                #image = localizeBoxes(image,[box])
                
    return image



def showSample(sample,thresh=0.5):
    
    image = sample["image"].numpy().transpose((1,2,0))
    annotations = sample["annotation"].numpy()
    
    image = localizeAnnotations(image,annotations,thresh)
    
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
    


def toDict(names,boxes,isGroundTruth = False):
    
    N,R,C,D = boxes.shape    
    boxes = boxes[:,:,:,:10].view((N,R*C,2,5)).reshape((N,R*C*2,5))
    
    dic = {}
    
    for n in range(N):
        for i in range(R*C*2):
            if names[n] not in dic:
                dic[names[n]] = {"bbox":[],"scores":[]}
            if (isGroundTruth and boxes[n,i,0] == 1) or not isGroundTruth:
                dic[names[n]]["scores"].append(boxes[n,i,0].tolist())
                dic[names[n]]["bbox"].append(boxes[n,i,1:].tolist())
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
        for score in pred_boxes[name]["scores"]:
            if score not in model_score.keys():
                model_score[score]=[name]
            else:
                model_score[score].append(name)
    return model_score 


    
    
    
def get_single_image_results(gt_boxes, pred_boxes, iou_thr):
    """Calculates number of true_pos, false_pos, false_neg from single batch of boxes.
    Args:
        gt_boxes 
        pred_boxes 
        iou_thr (float): value of IoU to consider as threshold for a
            true prediction.
    Returns:
        dict: true positives (int), false positives (int), false negatives (int)
    """
    real_bbox = torch.tensor(gt_boxes)
    pred_bbox = torch.tensor(pred_boxes)
    
    all_pred_indices = range(len(pred_bbox))
    all_gt_indices = range(len(real_bbox))
        
    if len(all_pred_indices)==0:
        tp=0
        fp=0
        fn=0
        return {'true_positive':tp, 'false_positive':fp, 'false_negative':fn}
    if len(all_gt_indices)==0:
        tp=0
        fp=0
        fn=0
        return {'true_positive':tp, 'false_positive':fp, 'false_negative':fn}
    
    gt_idx_thr=[]
    pred_idx_thr=[]
    ious=[]
    for ipb, pred_box in enumerate(pred_bbox):
        for igb, gt_box in enumerate(real_bbox):
            iou= iou_ybb(pred_box, gt_box)
            
            if iou >iou_thr:
                gt_idx_thr.append(igb)
                pred_idx_thr.append(ipb)
                ious.append(iou)
    iou_sort = np.argsort(ious)[::1]
    if len(iou_sort)==0:
        tp=0
        fp=0
        fn=0
        return {'true_positive':tp, 'false_positive':fp, 'false_negative':fn}
    else:
        gt_match_idx=[]
        pred_match_idx=[]
        for idx in iou_sort:
            gt_idx=gt_idx_thr[idx]
            pr_idx= pred_idx_thr[idx]
            # If the boxes are unmatched, add them to matches
            if(gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
                gt_match_idx.append(gt_idx)
                pred_match_idx.append(pr_idx)
        tp = len(gt_match_idx)
        fp = len(pred_bbox) - len(pred_match_idx)
        fn = len(real_bbox) - len(gt_match_idx)
    return {'true_positive': tp, 'false_positive': fp, 'false_negative': fn}    



def accuracy(true,preds,iou_thresh = 0.5):

    names = true.keys()
    
    TP = 0
    FP = 0
    FN = 0
    
    
    for name in names:
        
        true_bbox = true[name]["bbox"]
        pred_bbox = preds[name]["bbox"]
        
        res = get_single_image_results(true_bbox,pred_bbox,iou_thresh)
        
        TP+=res["true_positive"]
        FP+=res["false_positive"]
        FN+=res["false_negative"]
    
    return {"TP":TP,"FP":FP,"FN":FN}                    


def calc_precision_recall(results):
    
    TP = 0
    FP = 0
    FN = 0
    
    for ids in results.keys():
        
        res = results[ids]
        
        TP+=res["true_positive"]
        FP+=res["false_positive"]
        FN+=res["false_negative"]
    
    metrics = {"TP":TP,"FP":FP,"FN":FN} 
    
    prec = precision(metrics)
    rec = recall(metrics)
    
    return metrics,prec,rec
    
        

    
def precision(metrics):
    
    TP,FP,FN = metrics["TP"],metrics["FP"],metrics["FN"]  
    
    return TP/(TP+FP+eps)
    


def recall(metrics):
    
    TP,FP,FN = metrics["TP"],metrics["FP"],metrics["FN"]   
    
    return TP/(TP+FN+eps)


def F1(precision,recall):
    
    return 2*((precision*recall)/(precision + recall+eps))


def AP(gt_boxes, pred_boxes, iou_thr=0.5):
    
    model_scores = get_model_scores(pred_boxes)
    sorted_model_scores= sorted(model_scores.keys())
    
    names = gt_boxes.keys()
    
    # Sort the predicted boxes in descending order (lowest scoring boxes first):
    for name in names:
        
        arg_sort = np.argsort(pred_boxes[name]["scores"])
        pred_boxes[name]["scores"] = np.array(pred_boxes[name]["scores"])[arg_sort].tolist()
        pred_boxes[name]["bbox"] = np.array(pred_boxes[name]["bbox"])[arg_sort].tolist()

    pred_boxes_pruned = deepcopy(pred_boxes)
    precisions = []
    recalls = []
    model_thrs = []
    img_results = {}
    
    # Loop over model score thresholds and calculate precision, recall
    for ithr, model_score_thr in enumerate(sorted_model_scores[:-1]):
        
        # On first iteration, define img_results for the first time:
        img_ids = names if ithr == 0 else model_scores[model_score_thr]
        for name in img_ids:
               
            gt_boxes_img = gt_boxes[name]["bbox"]
            box_scores = pred_boxes[name]["scores"]
            start_idx = 0
            for score in box_scores:
                if score <= model_score_thr:
                    start_idx += 1
                else:
                    break 
                    
            #Remove boxes, scores of lower than threshold scores:
            pred_boxes_pruned[name]["bbox"]= pred_boxes_pruned[name]["bbox"][start_idx:]
            
            # Recalculate image results for this image
            img_results[name] = get_single_image_results(gt_boxes_img, pred_boxes_pruned[name]["bbox"], iou_thr=0.5)
            
        # calculate precision and recall
        metrics ,prec, rec = calc_precision_recall(img_results)
        precisions.append(prec)
        recalls.append(rec)
        model_thrs.append(model_score_thr)
        
        
        
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    prec_at_rec = []
    for recall_level in np.linspace(0.0, 1.0, 11):
        try:
            args= np.argwhere(recalls>recall_level).flatten()
            prec= max(precisions[args])
        except ValueError:
            prec=0.0
        prec_at_rec.append(prec)
    avg_prec = np.mean(prec_at_rec) 
    return {
        'ap': avg_prec,
        'precisions': precisions,
        'recalls': recalls,
        'thresh': model_thrs}



def mAP(true,preds,thresh):
    
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
    aps = 0.0
    
    # Iterate over data.
    for i,data in enumerate(tqdm(test_data)):
        
        names = data["name"]
        inputs = data["image"].to(device)
        labels = data["annotation"].to(device)

        # forward
        outputs =  model(inputs/255).view(labels.shape)
        loss = yoloLoss(labels.float(),outputs.float(),5,0.5)
        
        t_dict = toDict(names,labels,True)
        p_dict = toDict(names,outputs)
        
        ap = AP(t_dict,p_dict)["ap"]
        
        # statistics
        running_loss += loss.item()
        aps += ap
        print(f' Iteration Loss: {loss.item()} , Average Precision : {ap}')
                             
                    
    epoch_loss = running_loss / len(test_data)
    ap = aps / len(test_data)

    print(f" Final Loss: {epoch_loss} , Final Average Precision : {ap}")
          

    print()

    time_elapsed = (datetime.datetime.now() - since).total_seconds()
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
        
    return epoch_loss,ap

    
    