# -*- coding: utf-8 -*-

import tornado.ioloop
import tornado.web
import numpy as np
import tensorflow as tf
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow.python.framework import tensor_util
from PIL import Image
import base64
from io import StringIO
from io import BytesIO
import json

#默认的Tensorflow serving服务端IP和端口
TF_SERVING_HOST_PORT = "localhost:8500"
# #签名名称,与模型导出时设定的保持一致
SIGNATURE_NAME = "detection_signature"
# #默认的置信度阀值,超过阀值即视为识别到了目标
#DEFAULT_CONFIDENCE = 0.9




class PredictModel(tornado.web.RequestHandler):

    def load_image_into_numpy_array(self, image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)
	#def post(self):
        #res = json.loads(self.request.body)
        #self.write(json.dumps(res))

    def post(self):
        """ this is an echo...returns the same json document """
        # 创建一个logger
        print(self.request.body)
        print(self.request.body_arguments)
        b64_frame = self.request.body_arguments['b64_frame'][0]
        model_name="us"
        #model_name = self.request.body_arguments['name'][0]




        # b64_frame = b64_frame[2:-1]
        #binary_data = base64.b64encode(b64_frame)
        data = base64.b64decode(b64_frame)
        img_data = BytesIO(data)
		
		host, port = TF_SERVING_HOST_PORT.split(':')
        channel = implementations.insecure_channel(host, int(port))
        stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

        # Create prediction request object
        request = predict_pb2.PredictRequest()

        # Specify model name (must be the same as when the TensorFlow serving serving was started)
        request.model_spec.name = model_name
        image = Image.open(img_data)
        image_np = self.load_image_into_numpy_array(image)
        image_input = np.expand_dims(image_np, 0)
        # Initalize prediction
        # Specify signature name (should be the same as specified when exporting model)
        request.model_spec.signature_name = SIGNATURE_NAME
        request.inputs['inputs'].CopyFrom(
            tf.contrib.util.make_tensor_proto(image_input))

        #Call the prediction server
        result = stub.Predict(request, 10.0)  # 10 secs timeout

        #Plot boxes on the input image
        #category_index = self.load_label_map(FLAGS.path_to_labels)
        label = result.outputs['out'].float_val
        label =np.squeeze(label).tolist()
        # boxes = result.outputs['detection_boxes'].float_val
        # classes = result.outputs['detection_classes'].float_val
        # scores = result.outputs['detection_scores'].float_val
        # boxes = np.reshape(boxes, [100, 4]).tolist()
        # classes = np.squeeze(classes).astype(np.int32).tolist()
        # scores = np.squeeze(scores).tolist()
        #print scores
        # for i in range(len(scores)):
        #     if scores[i] < DEFAULT_CONFIDENCE :
        #         boxes=boxes[0:i]
		# 		classes=classes[0:i]
        #         scores=scores[0:i]
        #         break

        #ret = {'boxes': boxes, 'classes': classes, 'scores': scores}
        ret = dict()
        if model_name == "us" :
            ret["result"]="true"
            ret["label"]=label
        else:
            ret["result"]="false"




        response = json.dumps(ret)
        self.write(response)


application = tornado.web.Application([
  (r"/predict", PredictModel),
])
if __name__ == "__main__":
    try:
        print("Start the service")
        application.listen(8080)
        tornado.ioloop.IOLoop.instance().start()
    except KeyboardInterrupt:
        print("\nStop the service")
