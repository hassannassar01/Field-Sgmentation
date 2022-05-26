import os,time,cv2, sys, math
import tensorflow as tf
import argparse
import numpy as np

from utils import utils, helpers
from builders import model_builder

parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, default=None, required=True, help='The image you want to predict on. ')
parser.add_argument('--checkpoint_path', type=str, default=None, required=True, help='The path to the latest checkpoint weights for your model.')
parser.add_argument('--crop_height', type=int, default=1024, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=1024, help='Width of cropped input image to network')
parser.add_argument('--model', type=str, default=None, required=True, help='The model you are using')
parser.add_argument('--dataset', type=str, default="dataset", required=False, help='The dataset you are using')
args = parser.parse_args()

class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))

num_classes = len(label_values)

print("\n***** Begin prediction *****")
print("Dataset -->", args.dataset)
print("Model -->", args.model)
print("Crop Height -->", args.crop_height)
print("Crop Width -->", args.crop_width)
print("Num Classes -->", num_classes)

# Initializing network
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)

net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes]) 

network, _ = model_builder.build_model(args.model, net_input=net_input,
                                        num_classes=num_classes,
                                        crop_width=args.crop_width,
                                        crop_height=args.crop_height,
                                        is_training=False)

sess.run(tf.global_variables_initializer())

print('Loading model checkpoint weights')
saver=tf.train.Saver(max_to_keep=1000)
saver.restore(sess, args.checkpoint_path)


def demo():
  cap = cv2.VideoCapture(args.video)
  width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
  height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
  fps = cap.get(cv2.CAP_PROP_FPS)
  vid_writer = cv2.VideoWriter(
            'demo1.mp4', cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(1024), int(1024))
        )
  while True:
        ret_val, frame = cap.read()
        if ret_val:
          # loaded_image = utils.load_image(args.image)
          resized_image =cv2.resize(frame, (args.crop_width, args.crop_height))
          input_image = np.expand_dims(np.float32(resized_image),axis=0)/255.0
          
          output_image = sess.run(network,feed_dict={net_input:input_image})

          output_image = np.array(output_image[0,:,:,:])
          output_image = helpers.reverse_one_hot(output_image)
          
          out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
          out_vis_image = out_vis_image.astype(np.uint8)
          frame_copy = cv2.resize(frame,(1024, 1024))
          dst = cv2.addWeighted(out_vis_image,0.6,frame_copy,0.4,0)
          vid_writer.write(cv2.cvtColor(dst, cv2.COLOR_RGB2BGR))
          ch = cv2.waitKey(1)
          if ch == 27 or ch == ord("q") or ch == ord("Q"):
              break
        else:
            break
  cap.release()
  cv2.destroyAllWindows()
          # file_name = utils.filepath_to_name(args.image)
          # cv2.imwrite("%s_pred.png"%(file_name),cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))
st = time.time()
demo()
run_time = time.time()-st
print(500/run_time, "frames per secone")
print("Finished!")
# print("Wrote image " + "%s_pred.png"%(file_name))
