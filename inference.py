import tensorflow as tf
import numpy as np

threshold = 0.264895

def predict(df_fraud):
    model = tf.keras.models.load_model("model_best.keras")
    reconstcution_a=model.predict(df_fraud)
    reconstruction_a = np.array(reconstcution_a, dtype=np.float32)
    df_fraud = np.array(df_fraud, dtype=np.float32)
    test_loss=tf.keras.losses.mae(reconstruction_a,df_fraud)
    return tf.math.less(test_loss,threshold)