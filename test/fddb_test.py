import numpy as np
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

count = 0
Path = './examples/images/fddb_images/'
f = open('./test/fddb-dets.txt', 'wt')
for Name in open('./test/fddb_img_list.txt'):
    Image_Path = Path + Name[:-1] + '.jpg'
    image = caffe.io.load_image(Image_Path)
    im_scale = 3.0
    if im_scale != 1:
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

    f.write('{:s}\n'.format(Name[:-1]))
    f.write('{:.1f}\n'.format(det_conf.shape[0]))
    for i in xrange(det_conf.shape[0]):
        xmin = det_xmin[i] * image.shape[1]
        ymin = det_ymin[i] * image.shape[0]
        xmax = det_xmax[i] * image.shape[1]
        ymax = det_ymax[i] * image.shape[0]
        score = det_conf[i]
        f.write('{:.3f} {:.3f} {:.3f} {:.3f} {:.10f}\n'.
                format(xmin/im_scale, ymin/im_scale, (xmax-xmin+1)/im_scale, (ymax-ymin+1)/im_scale, score))
    count += 1
    print('%d' % count)
