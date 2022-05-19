import sys
import torch
import cv2
from utilsTorch import *
from model import *

def run(imagePath,backend = "mobilenet"):
    
    fileName = f"yolo-back-{backend}-weights"
   
    #Load model
    model = YOLOv1(backend)
    model.load_state_dict(torch.load(f"trained/{fileName}"))
    model.eval()
    
    #Read and preprocess the image.
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    processedImage = preProcessImage(image,(448,448))
    
    
    #Generate annotations.
    preds = model(processedImage.unsqueeze(0)/255).view(1,7,7,30).detach()
    sample = {"image":processedImage,"annotation":preds.squeeze(0)}
    
    #Show the annotated image.
    def onTrackBar(val):
        output = showSample(sample,val/10,False)
        cv2.imshow('Display', cv2.cvtColor(output,cv2.COLOR_BGR2RGB))

    cv2.namedWindow("Display", cv2.WINDOW_FREERATIO)

    cv2.createTrackbar("threshold", "Display" , 0, 10, onTrackBar)

    # Waiting 0ms for user to press any key
    cv2.waitKey(0)

    # Using cv2.destroyAllWindows() to destroy
    # all created windows open on screen
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    print(sys.argv)
    run(*sys.argv[1:])
    