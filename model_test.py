from keras_preprocessing.image import ImageDataGenerator, load_img
import matplotlib.pyplot as plt
import numpy as np
import model_helper

# Script for testing the model. Testing images should be be in subfolders of directory specified in test_path.
# Each subfolder name represents the name of the class.
test_path = "D:\Data\TEST"
trained_model_path = "model/test2-L2/model_arh2.json"
#trained_model_weights_path = "model/test1-dropout/model_w_103.h5"
trained_model_weights_path = "model/test2-L2/weights00000016.h5"
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

print("all predictions:",predictions)
top5 = []



predicted_classes = np.argmax(predictions, axis=1)
#print("predicted classes:",predicted_classes)

errors = np.where(predicted_classes != ground_truth)[0]
correct = np.where(predicted_classes == ground_truth)[0]
print("No of errors = {}/{}".format(len(errors), test_generator.samples))

for pred in predictions:
    top5.append(np.argsort(pred))

if len(errors) == 0:
    exit(0)

sum_err_conf = 0
for i in range(len(errors)):
    pred_class = np.argmax(predictions[errors[i]])
    pred_label = idx2label[pred_class]

    original_label = fnames[errors[i]].split('r\\')[0]
    conf = predictions[errors[i]][pred_class]
    conf_sorted = np.sort(predictions[errors[i]])
    top3 = np.flip(conf_sorted[-3:])
    sum_err_conf += top3[0]
    print("CLASS:",original_label)
    print("conf",conf,"conf3",top3)

sum_corr_conf = 0
for i in range(len(correct)):
    pred_class = np.argmax(predictions[correct[i]])
    pred_label = idx2label[pred_class]

    original_label = fnames[correct[i]].split('r\\')[0]
    conf = predictions[correct[i]][pred_class]
    conf_sorted = np.sort(predictions[correct[i]])
    top3 = np.flip(conf_sorted[-3:])
    sum_corr_conf += top3[0]
    print("CLASS:",original_label)
    print("conf",conf,"conf3",top3)

print("Avarage confidence of correctly predicted classes:",sum_corr_conf/(1.0*len(correct)))
print("Avarage confidence of incorrectly predicted classes:",sum_err_conf/(1.0*len(errors)))

"""
# Show the errors
for i in range(len(errors)):
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
"""