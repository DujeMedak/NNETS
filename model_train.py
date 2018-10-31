from keras import optimizers
from keras_preprocessing.image import ImageDataGenerator,load_img
import matplotlib.pyplot as plt
import numpy as np
import model_helper

# Script for training the model. Training and validation images should be be in subfolders of directory specified in paths below.
# Each subfolder name represents the name of the class.
train_dir = "D:\Data\TRAIN"
validation_dir = "D:\Data\TEST"

# saving paths
trained_model_path = "model_arh2.json"
trained_model_weights_path = "model_w_102.h5"

train_batchsize = 20
val_batchsize = 5
target_image_size = 128
num_of_epochs = 15

# change architecture in model helper if needed
model = model_helper.create_model(n_classes=25)

# TODO load all accuracy and loss infos from previous training if loading weights
#model.load_weights('small_last4.h5')

# Show a summary of the model. Check the number of trainable parameters
#model.summary()

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(target_image_size, target_image_size),
    batch_size=train_batchsize,
    class_mode='categorical')

validation_datagen = ImageDataGenerator(rescale=1. / 255)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(target_image_size, target_image_size),
    batch_size=val_batchsize,
    class_mode='categorical',
    shuffle=False)

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

# Train the model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples / train_generator.batch_size,
    epochs=num_of_epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples / validation_generator.batch_size,
    verbose=1)

# Save the model
model_helper.save_model_and_weights(model, saving_model_path=trained_model_path, weights_path = trained_model_weights_path)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

# Create a generator for prediction
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(target_image_size, target_image_size),
    batch_size=val_batchsize,
    class_mode='categorical',
    shuffle=False)

# Get the filenames from the generator
fnames = validation_generator.filenames

# Get the ground truth from generator
ground_truth = validation_generator.classes

# Get the label to class mapping from the generator
label2index = validation_generator.class_indices

# Getting the mapping from class index to class label
idx2label = dict((v, k) for k, v in label2index.items())

# Get the predictions from the model using the generator
predictions = model.predict_generator(validation_generator,
                                      steps=validation_generator.samples / validation_generator.batch_size, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)

errors = np.where(predicted_classes != ground_truth)[0]
print("No of errors = {}/{}".format(len(errors), validation_generator.samples))

print("saving to txt files (acc,values")
numpy_acc = np.array(acc)
np.savetxt("acc_history.txt", numpy_acc, delimiter=",")
numpy_acc_val = np.array(acc)
np.savetxt("acc_val_history.txt", numpy_acc_val, delimiter=",")

numpy_loss = np.array(acc)
np.savetxt("loss_history.txt", numpy_loss, delimiter=",")
numpy_loss_val = np.array(acc)
np.savetxt("loss_val_history.txt", numpy_loss_val, delimiter=",")

# Show the errors
for i in range(len(errors)):
    pred_class = np.argmax(predictions[errors[i]])
    pred_label = idx2label[pred_class]

    title = 'Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
        fnames[errors[i]].split('/')[0],
        pred_label,
        predictions[errors[i]][pred_class])

    original = load_img('{}/{}'.format(validation_dir, fnames[errors[i]]))
    plt.figure(figsize=[7, 7])
    plt.axis('off')
    plt.title(title)
    plt.imshow(original)
    plt.show()
