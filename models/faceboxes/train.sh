cd ../../
./build/tools/caffe train \
--solver="models/faceboxes/solver.prototxt" \
--gpu 0,1 2>&1 | tee models/faceboxes/train.log
