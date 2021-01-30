# IMAGE CLASSIFICATION

# 1. PULL  IN THE LIBRARIES ETC
# 2. LOAD THE DATA
# 3. EXPLORE THE DATA (FOLDER STRUCTURE INCLUDED)
# 4. MODEL DEFINITION
# 5. SET UP IMAGE DATA GENERATOR (TRAIN AND VALIDATION)
# 6. IMAGE AUGMENTATION
# 7. DEFINE CALLBACKS
# 8. MODEL COMPILE
# 9. MODEL TRAINING
# 10. PREDICTION

### FIX MY GPU SPACE ISSUES
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)

  except RuntimeError as e:
    print(e)


# GRADED FUNCTION: train_mnist_conv
def train_mnist_conv():
    # Please write your code only where you are indicated.
    # please do not remove model fitting inline comments.

    # YOUR CODE STARTS HERE
    ### DEFINE CALLBACK FOR EARLY STOPPING
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('accuracy') > 0.998):
                print("\nReached 99.8% accuracy so cancelling training!")
                self.model.stop_training = True

    # YOUR CODE ENDS HERE

    mnist = tf.keras.datasets.mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()

    # YOUR CODE STARTS HERE

    ### RESHAPE THE INPUT FOR CONVOLUTION TO SEE ALL THE IMAGES
    ### NORMALISE THE PIXEL VALUES
    training_images = training_images.reshape(60000, 28, 28, 1)
    training_images = training_images / 255.0
    test_images = test_images.reshape(10000, 28, 28, 1)
    test_images = test_images / 255.0

    callbacks = myCallback()
    # YOUR CODE ENDS HERE

    model = tf.keras.models.Sequential([
        # YOUR CODE STARTS HERE
        ### USE ONE CONV LAYER AND ONE MAX POOL
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
        # YOUR CODE ENDS HERE
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # model fitting
    history = model.fit(
        # YOUR CODE STARTS HERE
        ### UP TO 20 EPOCHS
        training_images, training_labels, epochs=20, callbacks=[callbacks]
        # YOUR CODE ENDS HERE
    )
    # model fitting
    return history.epoch, history.history['accuracy'][-1]


_, _ = train_mnist_conv()