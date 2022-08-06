import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import glob
import os
import cv2
import time
import numpy as np
from PIL import Image
import datetime
import pandas as pd


files = os.listdir("data")

PATH_TO_LABELS=r"\dataset\dataset2\hot_labelmap.pbtxt"
threshold=55

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

data_frame = pd.DataFrame(columns=['image', 'release', 'hold'])
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
    
PATH_TO_SAVED_MODEL = r"E:\ssd_hot_model\exported_model\saved_model"


print('Loading model...', end='')
start_time = time.time()

# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))

def rust_detection(img):
    lower_rust=(0,36,23)
    higher_rust=(33,255,255)
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    mask = cv2.inRange(hsv, lower_rust , higher_rust)
    cv2.imwrite("mask.jpg",mask)	
    percentage = (mask==255).mean() * 100

    mask_overlay=img.copy()
    mask_overlay[np.where((mask==[255]))] = (0,255,0)
    img = ((0.5 * img) + (0.5 * mask_overlay)).astype("uint8")  
    
    
    return percentage,img


#image_path="haha.jpg"


#('Running inference for {}... '.format(image_path), end='')
for i in range(len( files)):
    current_time=datetime.datetime.now()
    
    image_np=cv2.imread("data/"+files[i])
    img=cv2.imread("data/"+files[i])
    
    #image_np = load_image_into_numpy_array("data/"+files[i])
    basename = os.path.basename("data/"+files[i])
    
    ori_image = cv2.imread("data/"+files[i])
    
    # Things to try:
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()
    
    # Convert image to grayscale
    # image_np = np.tile(
    #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)
    
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]
    
    # input_tensor = np.expand_dims(image_np, 0)
    
    detections = detect_fn(input_tensor)
    
    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections
    
    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    
    image_np_with_detections = image_np.copy()
    
    filtered_detection=[i for i in range(len(detections['detection_scores'])) if detections['detection_scores'][i] >=0.3]
    
    
    for item in filtered_detection:

        minX,minY=int(detections['detection_boxes'][item][1]*image_np.shape[1]),int(detections['detection_boxes'][item][0]*image_np.shape[0])
        maxX,maxY=int(detections['detection_boxes'][item][3]*image_np.shape[1]),int(detections['detection_boxes'][item][2]*image_np.shape[0])
        
        defect_type=category_index[detections['detection_classes'][item]]["name"]
        
        percentage,ori_image[minY:maxY,minX:maxX]=rust_detection(ori_image[minY:maxY,minX:maxX])
        processed_time=(datetime.datetime.now()-current_time).microseconds / 1000
        if percentage >threshold:
    
            true = 1
            if "release" in files[i]:
                true = 0
                
            new_data = pd.DataFrame([{'image': files[i], 'release': "","hold":"Yes","True":true,"Process Duration":processed_time,"Image Resolution":(img.shape[1],img.shape[0]),"Number of Pixels":img.shape[1]*img.shape[0],"Rust Percentage":percentage}],
                               columns =['image', 'release', 'hold',"True","Process Duration","Image Resolution","Number of Pixels","Rust Percentage"])
            
            data_frame=data_frame.append(new_data, ignore_index = True)
            
            cv2.imwrite("hold/"+str(files[i]).split(".")[0]+".jpg",ori_image)
        else:
            true = 1
            if "hold" in files[i]:
                true = 0        
            cv2.imwrite("release/"+str(files[i]).split(".")[0]+".jpg",ori_image)
            new_data = pd.DataFrame([{'image': files[i], 'release': "Yes","hold":"","True":true,"Process Duration":processed_time,"Image Resolution":(img.shape[1],img.shape[0]),"Number of Pixels":img.shape[1]*img.shape[0],"Rust Percentage":percentage}],
                               columns =['image', 'release', 'hold',"True","Process Duration","Image Resolution","Number of Pixels","Rust Percentage"])
    
            data_frame=data_frame.append(new_data, ignore_index = True)
        
percentage_accuracy = len(data_frame[(data_frame['True']==1)])/len(data_frame) * 100

print("Accuracy:"+str(percentage_accuracy))
data_frame.to_excel("output.xlsx")  