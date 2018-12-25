import numpy as np
import matplotlib.pyplot as plt
import cv2,os,sys

caffe_root = '../'
os.chdir(caffe_root)
sys.path.insert(0, 'python')
import caffe
caffe.set_device(0)
caffe.set_mode_gpu()
model_def = 'models/faceboxes/deploy.prototxt'
model_weights = 'models/faceboxes/faceboxes.caffemodel'
net = caffe.Net(model_def, model_weights, caffe.TEST)

image = caffe.io.load_image('examples/images/1.jpg')
im_scale = 1.0
if im_scale != 1.0:
    image = cv2.resize(image, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
net.blobs['data'].reshape(1, 3, image.shape[0], image.shape[1])
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104, 117, 123])) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB
transformed_image = transformer.preprocess('data', image)
net.blobs['data'].data[...] = transformed_image

detections = net.forward()['detection_out']
det_label = detections[0, 0, :, 1]
det_conf = detections[0, 0, :, 2]
det_xmin = detections[0, 0, :, 3]
det_ymin = detections[0, 0, :, 4]
det_xmax = detections[0, 0, :, 5]
det_ymax = detections[0, 0, :, 6]

# Get detections with confidence higher than N.
top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.5]
top_conf = det_conf[top_indices]
top_xmin = det_xmin[top_indices]
top_ymin = det_ymin[top_indices]
top_xmax = det_xmax[top_indices]
top_ymax = det_ymax[top_indices]

##########Plot the boxes##########
plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
colors = plt.cm.hsv(np.linspace(0, 1, 2)).tolist()
plt.imshow(image)
currentAxis = plt.gca()
for i in xrange(top_conf.shape[0]):
    xmin = int(round(top_xmin[i] * image.shape[1]))
    ymin = int(round(top_ymin[i] * image.shape[0]))
    xmax = int(round(top_xmax[i] * image.shape[1]))
    ymax = int(round(top_ymax[i] * image.shape[0]))
    score = top_conf[i]
    display_txt = '%.2f' % (score)
    display_wh = '%.2f %.2f' % (xmax-xmin+1, ymax-ymin+1)
    coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
    color = colors[1]
    currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
    currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha': 0.5})
    currentAxis.text(xmin, ymax, display_wh, bbox={'facecolor': color, 'alpha': 0.5})
plt.show()
plt.close()
