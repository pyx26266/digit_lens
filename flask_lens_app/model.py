import tensorflow as tf
import cv2

sess = tf.Session()

saver = tf.train.import_meta_graph('session/model.ckpt.meta')
saver.restore(sess, tf.train.latest_checkpoint('./session/'))

graph = tf.get_default_graph()

features = graph.get_tensor_by_name('features:0')
Y_predict = graph.get_tensor_by_name('Y_predict:0')

def predict_from_image(img):
    img = img.reshape(-1, 28, 28, 1)
#     print(sess.run(Y_predict, feed_dict={features: img}))
    number = tf.argmax(Y_predict, 1)
    return int(sess.run(number, feed_dict={features: img}))

