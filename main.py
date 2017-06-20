import caffe

caffe_root = '/opt/project/'

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
model_file = caffe_root+'models/1miohands-modelzoo-v2/deploy.prototxt'
pretrained = caffe_root+'models/1miohands-modelzoo-v2/1miohands-v2.caffemodel'

caffe.set_mode_cpu()
net = caffe.Net (model_file,pretrained,caffe.TEST)

file1 ="data/images/final_phoenix_noPause_noCompound_lefthandtag_noClean/01August_2011_Monday_heute_default-6/1/*.png_fn000054-0.png"
file2 = "data/images/final_phoenix_noPause_noCompound_lefthandtag_noClean/01December_2011_Thursday_heute_default-3/1/*.png_fn000012-0.png"
file3= "data/images/final_phoenix_noPause_noCompound_lefthandtag_noClean/04July_2011_Monday_tagesschau_default-11/1/*.png_fn000053-0.png"

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))

net.blobs['data'].data[...] = transformer.preprocess('data',caffe.io.load_image(caffe_root+file1))
pred =net.forward()
print(pred)


net.blobs['data'].data[...] = transformer.preprocess('data',caffe.io.load_image(caffe_root+file2))
pred =net.forward()
print(pred)


net.blobs['data'].data[...] = transformer.preprocess('data',caffe.io.load_image(caffe_root+file3))
pred =net.forward()
print(pred)
