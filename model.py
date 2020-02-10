import tensorflow as tf
from tensorflow.keras.metrics import Recall, Precision, AUC, BinaryAccuracy
import tensorflow.keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

def nv1x16(n_kernels=128):
    l1 = 1e-9
    l2 = 1e-9

    loss = tf.keras.losses.binary_crossentropy

    model = tf.keras.Sequential(name='nv1x16')
    model.add(tf.keras.layers.BatchNormalization(input_shape=(6000, 16, 1)))
    model.add(tf.keras.layers.AveragePooling2D((2, 1)))

    model.add(tf.keras.layers.Conv2D(filters=n_kernels // 4,
                                     kernel_size=(5, 1),
                                     strides=(5, 1),
                                     padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l1_l2(l1, l2)
                                     ))
    model.add(tf.keras.layers.BatchNormalization())
    tf.keras.layers.LeakyReLU(0.2)
    model.add(tf.keras.layers.Dropout(.2))

    model.add(tf.keras.layers.Conv2D(filters=n_kernels // 2,
                                     kernel_size=(5, 1),
                                     padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l1_l2(l1, l2)))
    model.add(tf.keras.layers.BatchNormalization())
    tf.keras.layers.LeakyReLU(0.2)
    model.add(tf.keras.layers.MaxPooling2D((3, 1)))
    model.add(tf.keras.layers.Dropout(.2))

    for i in range(2):
        model.add(tf.keras.layers.Conv2D(filters=(n_kernels // 4) * (3 + i),
                                         kernel_size=(3, 1),
                                         padding='same',
                                         kernel_regularizer=tf.keras.regularizers.l1_l2(l1, l2)))
        model.add(tf.keras.layers.BatchNormalization())
        tf.keras.layers.LeakyReLU(0.2)
        model.add(tf.keras.layers.MaxPooling2D((2, 1)))
        model.add(tf.keras.layers.Dropout(0.2))

    n_k = {0: n_kernels,
           1: (n_kernels // 4) * 3,
           2: n_kernels // 2,
           3: n_kernels // 4,
           4: n_kernels // 4,
           5: n_kernels // 4,
           }

    for i in range(3):
        model.add(tf.keras.layers.Conv2D(filters=n_k[2*i],
                                         kernel_size=(4, 1),
                                         padding='valid',
                                         kernel_regularizer=tf.keras.regularizers.l1_l2(l1, l2)))
        model.add(tf.keras.layers.Conv2D(filters=n_k[2*i+1],
                                         kernel_size=(4, 1),
                                         padding='valid',
                                         kernel_regularizer=tf.keras.regularizers.l1_l2(l1, l2)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU(0.2))
        model.add(tf.keras.layers.MaxPooling2D((2, 1)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(.5))
    model.add(tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l1_l2(l1, l2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(loss=loss,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=[AUC(curve='ROC', name='roc_auc'), AUC(curve='PR', name='pr_auc'), Precision(), Recall(), mean_pred, BinaryAccuracy()])

    return model