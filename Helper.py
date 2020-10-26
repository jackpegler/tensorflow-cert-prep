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









################ MODEL SAVING AND LOADING ################

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
