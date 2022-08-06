from copy import copy
from urllib import response
from django.shortcuts import render
# Create your views here.

from django.core.files.storage import FileSystemStorage

###### dependies of your model #############
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
#from tf.keras.preprocessing import image
import glob
import os
import cv2
import time
import numpy as np
from PIL import Image
import datetime
import pandas as pd
import xlwt
from django.http import HttpResponse


img_height, img_width=224,224
PATH_TO_LABELS= r"/Users/neerajsingh/Desktop/django_project/ImageClassification_DjangoApp-master/model/hot_labelmap.pbtxt"
threshold=55

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

data_frame = pd.DataFrame(columns=['image', 'release', 'hold'])
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


print("step 1")



PATH_TO_SAVED_MODEL = r"/Users/neerajsingh/Desktop/django_project/ImageClassification_DjangoApp-master/model/saved_model"


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



directory = r"/Users/neerajsingh/Desktop/django_project/model/data"
IMAGE_PATHS = [directory + "/" + f for f in os.listdir(directory) if f[-4:] in ['.jpg','.png','.bmp']]


def final_predict (image_path):
    current_time=datetime.datetime.now()
    data_frame = pd.DataFrame()
    image_np=cv2.imread(image_path)
    img=cv2.imread(image_path)
    
    image_np = load_image_into_numpy_array(image_path)
    basename = os.path.basename(image_path)
    
    ori_image = cv2.imread(image_path)
    
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
            if "release" in image_path:
                true = 0
                
            new_data = pd.DataFrame([{'image': image_path, 'release': "","hold":"Yes","true":true,"Process_Duration":processed_time,"Image_Resolution":(img.shape[1],img.shape[0]),"Number_of_Pixels":img.shape[1]*img.shape[0],"Rust_Percentage":percentage}],
                               columns =['image', 'release', 'hold',"true","Process_Duration","Image_Resolution","Number_of_Pixels","Rust_Percentage"])
            
            #data_frame=data_frame.append(new_data, ignore_index = True)
            #print("hold")
            msg = "hold"
            cv2.imwrite("hold/"+str(image_path).split(".")[0]+".jpg",ori_image)
        else:
            true = 1
            if "hold" in image_path:
                true = 0        
            cv2.imwrite("release/"+str(image_path).split(".")[0]+".jpg",ori_image)
            new_data = pd.DataFrame([{'image': image_path, 'release': "Yes","hold":"","true":true,"Process_Duration":processed_time,"Image_Resolution":(img.shape[1],img.shape[0]),"Number_of_Pixels":img.shape[1]*img.shape[0],"Rust_Percentage":percentage}],
                               columns =['image', 'release', 'hold',"true","Process_Duration","Image_Resolution","Number_of Pixels","Rust_Percentage"])
    
            #data_frame=data_frame.append(new_data, ignore_index = True)
            msg = "released"
            #print("released")
        return (msg , new_data)




def index(request):
    context={'a':1}
    return render(request,'index.html',context)


data_frame = pd.DataFrame()

def predictImage(request):
    #df = pd.DataFrame()
    global data_frame
    print (request)
    print (request.POST.dict())
    fileObj=request.FILES['filePath']
    fs=FileSystemStorage()
    filePathName=fs.save(fileObj.name,fileObj)
    filePathName=fs.url(filePathName)
    testimage='.'+filePathName
    img = Image.open(testimage)
    msg = final_predict (testimage)[0]
    data_frame = data_frame.append(final_predict(testimage)[1], ignore_index = True)
    #x = image.img_to_array(img)
    #x=x/255
    #x=x.reshape(1,img_height, img_width,3)
   # with model_graph.as_default():
    #    with tf_session.as_default():
     #       predi=model.predict(x)

    #import numpy as np
    #predictedLabel=labelInfo[str(np.argmax(predi[0]))]
    #filePathName=fs.save(final_predict(testimage)[1][['image']],fileObj)
    context={'filePathName':filePathName,'predictedLabel':msg , 'final_image_name' :final_predict(testimage)[1][['image']]}
    return render(request,'index.html',context) 

def viewDataBase(request):
    import os
    global data_frame
    listOfImages=os.listdir('./media/')
    listOfImagesPath=['./media/'+i for i in listOfImages]
    context={'listOfImagesPath':listOfImagesPath}
    data_frame.to_excel("output.xlsx") 
    return render(request,'viewDB.html',context) 


def download_file(request):
    pass    


#from django.http import HttpResponse
#from io import BytesIO

''''
def viewDataBase(request):
    import os
    listOfImages=os.listdir('./media/')
    listOfImagesPath=['./media/'+i for i in listOfImages]
    context={'listOfImagesPath':listOfImagesPath}
    data_frame.to_excel("output.xlsx") 
    df = pd.DataFrame()
    with BytesIO() as b:
        # Use the StringIO object as the filehandle.
        writer = pd.ExcelWriter(b, engine='xlsxwriter')
        df.to_excel(writer, sheet_name='Sheet1')
        writer.save()
        # Set up the Http response.
        filename = 'output.xlsx'
        #filename = 'output.xlsx'
        response = HttpResponse(
            b.getvalue(),
            content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        response['Content-Disposition'] = 'attachment; filename=%s' % filename
        return render(response , request, 'viewDB.html',context)

  '''      




'''
def download_excel_data(request):
	# content-type of response
	response = HttpResponse(content_type='application/ms-excel')

	#decide file name
	response['Content-Disposition'] = 'attachment; filename="ThePythonDjango.xls"'

	#creating workbook
	wb = xlwt.Workbook(encoding='utf-8')

	#adding sheet
	ws = wb.add_sheet("sheet1")

	# Sheet header, first row
	row_num = 0

	font_style = xlwt.XFStyle()
	# headers are bold
	font_style.font.bold = True

	#column header names, you can use your own headers here
	columns = ['Column 1', 'Column 2', 'Column 3', 'Column 4', ]

	#write column headers in sheet
	for col_num in range(len(columns)):
		ws.write(row_num, col_num, columns[col_num], font_style)

	# Sheet body, remaining rows
	font_style = xlwt.XFStyle()

	#get your data, from database or from a text file...
	data = get_data() #dummy method to fetch data.
	for my_row in data:
		row_num = row_num + 1
		ws.write(row_num, 0, my_row.name, font_style)
		ws.write(row_num, 1, my_row.start_date_time, font_style)
		ws.write(row_num, 2, my_row.end_date_time, font_style)
		ws.write(row_num, 3, my_row.notes, font_style)

	wb.save(response)
	return response

'''


def download_excel_data(request):
	# content-type of response
  response = HttpResponse(content_type='application/ms-excel')
  global data_frame
  #data_frame = pd.read_excel(r"/Users/neerajsingh/Desktop/django_project/ImageClassification_DjangoApp-master/output.xlsx")

	#decide file name
  response['Content-Disposition'] = 'attachment; filename="ThePythonDjango.xls"'

	#creating workbook
  wb = xlwt.Workbook(encoding='utf-8')

	#adding sheet
  ws = wb.add_sheet("sheet1")

	# Sheet header, first row
  row_num = 0

  font_style = xlwt.XFStyle()
	# headers are bold
  font_style.font.bold = True

	#column header names, you can use your own headers here
  columns = ['image', 'release', 'hold',"True","Process Duration","Number of Pixels","Rust Percentage","Image Resolution", ]

	#write column headers in sheet
  for col_num in range(len(columns)):
	  ws.write(row_num, col_num, columns[col_num], font_style)

	# Sheet body, remaining rows
  font_style = xlwt.XFStyle()

	#get your data, from database or from a text file...
  data = data_frame.reset_index() #dummy method to fetch data.
  for index, my_row in data.iterrows():
    row_num = row_num + 1
    ws.write(row_num, 0, my_row['image'], font_style)
    ws.write(row_num, 1, my_row['release'], font_style)
    ws.write(row_num, 2, my_row['hold'], font_style)
    ws.write(row_num, 3, my_row['true'], font_style)
    ws.write(row_num, 4, my_row['Process_Duration'], font_style)
    ws.write(row_num, 5, my_row['Number_of_Pixels'], font_style)
    ws.write(row_num, 6, my_row['Rust_Percentage'], font_style)
    ws.write(row_num, 7, str(my_row['Image_Resolution']), font_style)


  wb.save(response)
  return response



