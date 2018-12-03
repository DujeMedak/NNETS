from keras import models
from keras.applications import VGG16
from keras.engine.saving import model_from_json
from keras import regularizers
from keras import layers

# Script that contains methods for easier model manipulation (creating, savingf)
def save_model_and_weights(trained_model, saving_model_path, weights_path):
    trained_model.save(weights_path)
    model_json = trained_model.to_json()
    with open(saving_model_path, "w") as json_file:
        json_file.write(model_json)
    print("Saved model and weights to disk")


def save_model__weights(trained_model, weights_path):
    trained_model.save(weights_path)
    print("Saved model weights to disk in directory:", weights_path)


def load_model_from_file(model_path):
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    print("Loaded model from disk")
    return loaded_model


def create_model(n_classes,target_image_size=128, n_last_layers_to_freeze=4):
    # Load the VGG model
    vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(target_image_size, target_image_size, 3))
    # Freeze the layers except the last 4 layers
    for layer in vgg_conv.layers[:-n_last_layers_to_freeze]:
        layer.trainable = False

    # Create the complete model
    model = models.Sequential()
    # Add the vgg convolution base model
    model.add(vgg_conv)
    # Add new layers
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(n_classes, activation='softmax'))
    return model
