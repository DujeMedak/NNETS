from keras_preprocessing.image import ImageDataGenerator, load_img
import matplotlib.pyplot as plt
import numpy as np
import model_helper
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# Script for testing the model. Testing images should be be in subfolders of directory specified in test_path.
# Each subfolder name represents the name of the class.
test_path = "C:/Users/Marko/Desktop/Neuronske Projekt/NNETS-master/Data/TEST"
trained_model_path = "C:/Users/Marko/Desktop/Neuronske Projekt/model_arh2.json"
#trained_model_weights_path = "model/test1-dropout/model_w_103.h5"
trained_model_weights_path = "C:/Users/Marko/Desktop/Neuronske Projekt/weights00000016.h5"
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
print("all predictions:", predictions)
class_names = ['Airport', 'Beach', 'Bridge', 'Center', 'Church',
           'DenseResidential', 'Farmland', 'Forest', 'Industrial',
           'Meadow', 'MediumResidential', 'Mountain', 'Park', 'Parking',
           'Playground', 'Pond', 'Port', 'RailwayStation', 'River',
           'School', 'SparseResidential', 'Square', 'Stadium',
           'StorageTanks', 'Viaduct']
confusion = confusion_matrix(ground_truth, predicted_classes)
np.set_printoptions(precision=2)
plt.figure(figsize=[10, 10])
plot_confusion_matrix(confusion, classes=class_names,
                      title='Confusion matrix, without normalization')
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 3}
plt.figure(figsize=[15, 15])

plot_confusion_matrix(confusion, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
top5 = []
truth_counter = 0
for pred in predictions:
    picture_counter = truth_counter+1
    currenttruth = ground_truth[truth_counter]
    print(picture_counter, 'slika prikazuje ', class_names[currenttruth])
    sortedargument = np.argsort(pred)
    top_score = 24
    i = 0
    truth_counter=truth_counter+1
    while i < 5:
        i = i+1
        print(top_score)
        currentbest = sortedargument[top_score]
        print(i, '   ', class_names[currentbest])
        top_score = top_score-1
f1macro = f1_score(ground_truth, predicted_classes, average='macro')
f1micro = f1_score(ground_truth, predicted_classes, average='micro')
f1weighted = f1_score(ground_truth, predicted_classes, average='weighted')
f1none = f1_score(ground_truth, predicted_classes, average=None)
print('f1 macro', f1macro)
print('f1 micro', f1micro)
print('f1 weighted', f1weighted)
print('f1 none', f1none)