from keras_preprocessing.image import ImageDataGenerator,load_img
import matplotlib.pyplot as plt
import numpy as np
import model_helper

# Script for testing the model. Testing images should be be in subfolders of directory specified in test_path.
# Each subfolder name represents the name of the class.
test_path = "D:/Data/HR"
trained_model_path = "model_arh2.json"
trained_model_weights_path = "model_w_102.h5"
target_image_size = 128
test_batchsize = 1


model = model_helper.load_model_from_file(trained_model_path)
model.load_weights(trained_model_weights_path)

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(target_image_size, target_image_size),
    batch_size=test_batchsize,
    class_mode='categorical',
    shuffle=False)

# Get the filenames from the generator
fnames = test_generator.filenames

# Get the ground truth from generator
ground_truth = test_generator.classes

# Get the label to class mapping from the generator
label2index = test_generator.class_indices

# Getting the mapping from class index to class label
idx2label = dict((v, k) for k, v in label2index.items())

# Get the predictions from the model using the generator
predictions = model.predict_generator(test_generator,
                                      steps=test_generator.samples / test_generator.batch_size, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)

errors = np.where(predicted_classes == ground_truth)[0]

print("No of errors = {}/{}".format(len(errors), test_generator.samples))

if len(errors) == 0:
    exit(0)

# Show the errors
for i in range(len(predictions)):
    pred_class = np.argmax(predictions[errors[i]])
    pred_label = idx2label[pred_class]

    title = 'Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
        fnames[errors[i]].split('/')[0],
        pred_label,
        predictions[errors[i]][pred_class])

    original = load_img('{}/{}'.format(test_path, fnames[errors[i]]))
    plt.figure(figsize=[7, 7])
    plt.axis('off')
    plt.title(title)
    plt.imshow(original)
    plt.show()