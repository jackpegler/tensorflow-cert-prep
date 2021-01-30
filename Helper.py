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

## BINARY - SIGMOID, BINARY LOSS + IN DATAGEN
## MULIT-CLASS - SOFTMAX, CATEGORICAL LOSS + IN DATAGEN

################ DATA SETS ################
#---------------------------------------------


######### LOAD DATASETS FROM TF/KERAS
mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

### RESHAPE THE INPUT FOR CONVOLUTION TO SEE ALL THE IMAGES
### NORMALISE THE PIXEL VALUES
training_images = training_images.reshape(60000, 28, 28, 1)
training_images = training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images / 255.0




###### CREATE TRAIN AND TEST FOLDER STRUCTURE
try:
    os.mkdir('/tmp/cats-v-dogs/')
    os.mkdir('/tmp/cats-v-dogs/training/')
    os.mkdir('/tmp/cats-v-dogs/testing/')
    os.mkdir('/tmp/cats-v-dogs/training/cats/')
    os.mkdir('/tmp/cats-v-dogs/testing/cats/')
    os.mkdir('/tmp/cats-v-dogs/training/dogs/')
    os.mkdir('/tmp/cats-v-dogs/testing/dogs/')
except OSError:
    pass


def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    # YOUR CODE STARTS HERE
    source_files = os.listdir(SOURCE)
    source_files = random.sample(source_files, len(source_files))
    split_index = int(len(source_files) * SPLIT_SIZE)

    train_files = source_files[:split_index]
    test_files = source_files[split_index:]

    for file in train_files:
        if os.path.getsize(SOURCE + file) > 0:
            copyfile(SOURCE + file, TRAINING + file)

    for file in test_files:
        if os.path.getsize(SOURCE + file) > 0:
            copyfile(SOURCE + file, TESTING + file)


CAT_SOURCE_DIR = "/tmp/PetImages/Cat/"
TRAINING_CATS_DIR = "/tmp/cats-v-dogs/training/cats/"
TESTING_CATS_DIR = "/tmp/cats-v-dogs/testing/cats/"
DOG_SOURCE_DIR = "/tmp/PetImages/Dog/"
TRAINING_DOGS_DIR = "/tmp/cats-v-dogs/training/dogs/"
TESTING_DOGS_DIR = "/tmp/cats-v-dogs/testing/dogs/"

split_size = .9
split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)

################ BUILD SEQUENTIAL MODEL ################
#------------------------------------------------
# 2D CONV AND POOLING
#------------------------------------------------
model = tf.keras.models.Sequential([
    # YOUR CODE HERE
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])



################ IMAGE DATA GENERATOR (FLOW FROM DIRECTORY) ################

## TRAIN CAN PULL IN SOME IMAGE AUGMENTATION TO PREVENT OVER FITTTING
train_datagen = ImageDataGenerator(
                      rescale=1./255,
                      rotation_range=40,
                      width_shift_range=0.2,
                      height_shift_range=0.2,
                      shear_range=0.2,
                      zoom_range=0.2,
                      horizontal_flip=True,
                      fill_mode='nearest')


## SET LOCATION OF IMAGES, BATCH SIZE AND CLASS AND TARGET SIZE
train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size=10,
                                                    class_mode='binary',
                                                    target_size=(150, 150))


## TEST / VAL DON'T AUGMENT
validation_datagen = ImageDataGenerator(rescale = 1.0/255)

## SAME BATCH ETC
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                    batch_size=10,
                                                    class_mode='binary',
                                                    target_size=(150, 150))

## CAN USE SAME DATAGEN FOR TEST AND VAL, JUST CALL ON RELEVANT LOCATION



################ COMPILE MODEL ################
model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])






################ CUSTOM CALL BACKS ################
### DEFINE CALLBACK FOR EARLY STOPPING
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('accuracy') > 0.998):
                print("\nReached 99.8% accuracy so cancelling training!")
                self.model.stop_training = True

################ BUILT IN CALL BACKS ################
# using early stopping to exit training if validation loss is not decreasing even after certain epochs (patience)
earlystopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

# save the best model with lower validation loss
checkpointer = ModelCheckpoint(filepath="weights.hdf5", verbose=1, save_best_only=True)






################ TRAIN THE MODEL ################
history = model.fit_generator(train_generator,
                              epochs=2,
                              verbose=1,
                              validation_data=validation_generator,
                              callbacks=[callbacks]
                              )



################ PLOT THE ACCURACY AND LOSS ################

acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")


plt.title('Training and validation loss')



#------------------------------------------------
#  TRANSFER LEARNING
#------------------------------------------------
################ LOAD MODELS ################

## CAN ALSO LOAD WITH PRE-TRAINED WEIGHTS
# Import the inception model
from tensorflow.keras.applications.inception_v3 import InceptionV3

# Create an instance of the inception model from the local pre-trained weights
local_weights_file = path_inception

# ALTERNATIVE TO LOAD WITH WEIGHTS weights = 'imagenet'
pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
                                include_top=False,
                                weights=None)

pre_trained_model.load_weights(local_weights_file)

# Make all the layers in the pre-trained model non-trainable
for layer in pre_trained_model.layers:
    # Your Code Here
    layer.trainable = False

# Print the model summary
pre_trained_model.summary()


################ STACK NEW LAYERS ################
last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output


# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)
# Add a final sigmoid layer for classification
x = layers.Dense(1, activation='sigmoid')(x)

model = Model(pre_trained_model.input, x)


##### methoD 2
# Freeze the basemodel weights , so these weights won't change during training
basemodel.trainable = False
# Add classification head to the model
headmodel = basemodel.output
headmodel = GlobalAveragePooling2D(name = 'global_average_pool')(headmodel)
headmodel = Flatten(name='flatten')(headmodel)
headmodel = Dense(256, activation='relu', name='dense_1')(headmodel)
headmodel = Dropout(0.3)(headmodel)
headmodel = Dense(128, activation='relu', name='dense_2')(headmodel)
headmodel = Dropout(0.3)(headmodel)
headmodel = Dense(11, activation = 'softmax', name='dense_3')(headmodel)

model = Model(inputs = basemodel.input, outputs = headmodel)






################ MODEL SAVING AND LOADING ################
#---------------------------------------------

### MODEL SAVING
model.save('path/to/location')

### LOAD MODEL
model = tf.keras.models.load_model('path/to/location')

### MODEL FILE SIZE
import os
# Get file size in bytes for a given model
os.stat('model.h5').st_size

### MODEL PARAMETERS SIZE
model.summary()
