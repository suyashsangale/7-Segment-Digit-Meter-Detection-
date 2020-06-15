import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from object_detection.utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util

from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2

from numpy import loadtxt
from keras.models import load_model
import os

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')
#%matplotlib inline
PATH_TO_FROZEN_GRAPH = r'Capstone_project_2\inference_graph\frozen_inference_graph.pb'
PATH_TO_LABELS = r'Capstone_project_2\training\labelmap.pbtxt'

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
    
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def run_inference_for_single_image(image, graph):
    if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    # Run inference
    output_dict = sess.run(tensor_dict,
                            feed_dict={image_tensor: np.expand_dims(image, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict

image_name = sys.argv[1]	
input_image_file_name = 'no_flash_meter_Copy/'+ image_name
print(input_image_file_name)



os.getcwd()
#os.chdir(r'C:\Users\Suyash\no_flash_meter_Copy')

with detection_graph.as_default():
        with tf.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                  'num_detections', 'detection_boxes', 'detection_scores',
                  'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                      tensor_name)
                image_np = cv2.imread(input_image_file_name) #m6
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                output_dict = run_inference_for_single_image(image_np, detection_graph)
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        output_dict['detection_boxes'],
                        output_dict['detection_classes'],
                        output_dict['detection_scores'],
                        category_index,
                        instance_masks=output_dict.get('detection_masks'),
                        use_normalized_coordinates=True,
                        line_thickness=0)
                cv2.imshow('object_detection', cv2.resize(image_np, (800, 600)))
                cv2.waitKey(0)
                cv2.destroyAllWindows()

coordinate = []
# This is the way I'm getting my coordinates
boxes = output_dict['detection_boxes']
# get all boxes from an array
max_boxes_to_draw = boxes.shape[0]
# get scores to get a threshold
scores = output_dict['detection_scores']
# this is set as a default but feel free to adjust it to your needs
min_score_thresh=.5
# iterate over all objects found
for i in range(min(max_boxes_to_draw, boxes.shape[0])):
    # 
    if scores is None or scores[i] > min_score_thresh:
        # boxes[i] is the box which will be drawn
        class_name = category_index[output_dict['detection_classes'][i]]['name']
        print ("This box is gonna get used", boxes[i], output_dict['detection_classes'][i])
        coordinate=boxes[i]
        

height, width, channel = image_np.shape
ymin= int((coordinate[0])*height)
xmin= int((coordinate[1])*width)
ymax = int((coordinate[2])*height)

xmax = int((coordinate[3])*width)


crop_img = image_np[ymin:ymax, xmin:xmax]
crop_img = cv2.resize(crop_img,(600,100))
cv2.imshow("cropped", crop_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


gray = cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)
cv2.imshow('gray',gray)
cv2.waitKey(0)
cv2.destroyAllWindows()


def crop_imgage(img, scale=1.0):
    center_x, center_y = img.shape[1] / 2, img.shape[0] / 2
    width_scaled, height_scaled = img.shape[1] * scale, img.shape[0] * scale
    left_x, right_x = center_x - width_scaled / 2, center_x + width_scaled / 2
    top_y, bottom_y = center_y - height_scaled / 2, center_y + height_scaled / 2
    img_cropped = img[int(top_y):int(bottom_y), int(left_x):int(right_x)]
    return img_cropped

crop_img = crop_imgage(crop_img, 0.90)

cv2.imshow('bilateral',crop_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#blur = cv2.GaussianBlur(gray,(5,5),1)

blur = cv2.fastNlMeansDenoisingColored(crop_img,None,10,10,7,3)
blur = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
#blur= clahe.apply(gray)
thresh = cv2.adaptiveThreshold(blur,255,1,1,17,3)
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
dst = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
dst = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
cannny =cv2.Canny(blur, 10, 25)

cv2.imshow('thresh',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

threshold2 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
threshold3 = cv2.morphologyEx(threshold2, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

cv2.imshow('thresh1',threshold3)
cv2.waitKey(0)
cv2.destroyAllWindows()


threshold3 = cv2.morphologyEx(dst, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
#threshold2 = cv2.dilate(threshold2, np.ones((5,1), np.uint8), iterations=1)
height, width = threshold2.shape[:2]
threshold2 = threshold2[5:height,5:width]


cv2.imshow('thresh_3',threshold3)
cv2.waitKey(0)
cv2.destroyAllWindows()

dilation = cv2.dilate(threshold3,kernel,iterations = 2)

cv2.imshow('dialate',dilation)
cv2.waitKey(0)
cv2.destroyAllWindows()

im = dilation.copy()


model = load_model('model_2.h5')
model.summary()

# import the necessary packages

cnts = cv2.findContours(im.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
digitCnts = []
 
# loop over the digit area candidates
for c in cnts:
    # compute the bounding box of the contour
    (x, y, w, h) = cv2.boundingRect(c)
 
    # if the contour is sufficiently large, it must be a digit
    if w >= 0 and (h >= 35):
        digitCnts.append(c)

print("Number of Digits present is "+str(len(digitCnts)))
img_rows, img_cols = 48,68

digitCnts = contours.sort_contours(digitCnts,method="left-to-right")[0]
digits = []
op_dig = []
#dim = (48,68)

for i in range(0,len(digitCnts)):
    mask = np.zeros_like(im) # Create mask where white is what we want, black otherwise
    cv2.drawContours(mask, digitCnts, i, 255, -1) # Draw filled contour in mask
    out = np.zeros_like(im) # Extract out the object and place into output image
    out[mask == 255] = im[mask == 255]
    (y, x) = np.where(mask == 255)
    (topy, topx) = (np.min(y), np.min(x))
    (bottomy, bottomx) = (np.max(y), np.max(x))
    out = out[topy:bottomy+1, topx:bottomx+1]
    #out = cv2.erode(out, kernel, iterations=1) 
    #os.chdir(r'digits')
    print(out.shape)
    cv2.imshow('Output', out)
    output = out.copy()
    im_1 = cv2.resize(output,  (img_rows, img_cols)) 
    im_1.reshape((img_rows,img_cols))
    #print(im_1.shape) # (28,28)
    batch = np.expand_dims(im_1,axis=0)
    batch = np.expand_dims(batch,axis=3)
    pr = model.predict_classes(im_1.reshape((1, 68, 48,1)))
    print(pr)
    s = pr[0]
    op_dig.append(s)
    pro = model.predict_proba(im_1.reshape((1, 68, 48,1)))
    #print(pro)
    #print(pro[0,s])
    ##cv2.imwrite("template {0}.jpg".format(i),out)
    #print(pytesseract.image_to_string(out, lang="letsgodigital", config="--psm 10 -c tessedit_char_whitelist=.0123456789"))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#os.chdir(r'C:\Users\Suyash')

if(op_dig[0]!=0)and (len(digitCnts)==9):
    op_dig.pop(0)
	
print("Predicted Digits are :")
print(op_dig)
readings = ''.join(map(str, op_dig))
print("Meter Reading is " + readings + " units")
cv2.imshow('object_detection', cv2.resize(image_np, (800, 600)))
cv2.waitKey(0)
cv2.destroyAllWindows()
