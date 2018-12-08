from keras_preprocessing.image import ImageDataGenerator
import numpy as np
import model_helper
from PIL import Image, ImageDraw, ImageFont
import os
import math


##################LOADING THE IMAGE#############################################

def divide_and_conquer(map_file_path, output_map_sections_path,
                       target_section_width = 128,
                       target_section_height = 128,
                       class_subfolder = 'unclassified',
                       sections_file_format = 'png'):
    
    whole_map = Image.open(map_file_path)

    output_path = output_map_sections_path + '/' + class_subfolder + '/'

    # Create output directory if it doesnt exitst.
    os.makedirs(output_path, exist_ok = True)

    n_sections_width =                                                          \
        n_sections_in_dim(whole_map.width, target_section_width)
    n_sections_height =                                                         \
        n_sections_in_dim(whole_map.height, target_section_height)
    
    offset_width =                                                              \
        calc_intersection_offset(whole_map.width, target_section_width)
    offset_height =                                                             \
        calc_intersection_offset(whole_map.height, target_section_height)

    print(n_sections_width)
    print(n_sections_height)
    print(offset_width)
    print(offset_height)

    for i in range(n_sections_height):
        for j in range(n_sections_width):
            region = whole_map.crop((j*offset_width, i*offset_height,
                                    j*offset_width + target_section_width,
                                    i*offset_height + target_section_height))
            region.save(output_path +
                        str(i*n_sections_width + j) +
                        '.' +
                        sections_file_format)



def load_sections(output_map_sections_path,
                  test_batchsize = 1,
                  target_section_width = 128,
                  target_section_height = 128):

    tf_image_data_gen = ImageDataGenerator(rescale=1. / 255)

    tf_image_data = tf_image_data_gen.flow_from_directory(
        output_map_sections_path,
        target_size = (target_section_height, target_section_width),
        class_mode = None,
        batch_size = test_batchsize,
        shuffle = False,
        save_to_dir='./Data/my_test'
    )

    return tf_image_data

def n_sections_in_dim(img_dim_size, section_dim_size):
    return math.ceil(img_dim_size/section_dim_size)

# dim in the names stands for 'dimension' because this function is used for
# both width and height.
def calc_intersection_offset(img_dim_size, section_dim_size):
    n_sections = n_sections_in_dim(img_dim_size, section_dim_size)

    last_section_pos = img_dim_size - section_dim_size

    if n_sections == 1:
        return 0
    else:
        return last_section_pos/(n_sections - 1)


##########################NET PREDICTION########################################

def get_predictions(img_data, model_file_path, weights_file_path):

    model = model_helper.load_model_from_file(model_file_path)
    model.load_weights(weights_file_path)

    print(img_data.samples)

    # Get the predictions from the model using the generator
    predictions = model.predict_generator(img_data,
                                        steps=img_data.samples / img_data.batch_size, verbose=1)

    return predictions


###########################DISPLAY PREDICTIONS##################################


def get_box_from_index(index, full_width, full_height,
                       target_section_width = 128,
                       target_section_height = 128):

    n_sections_width =                                                          \
        n_sections_in_dim(full_width, target_section_width)
    
    offset_width =                                                              \
        calc_intersection_offset(full_width, target_section_width)
    offset_height =                                                             \
        calc_intersection_offset(full_height, target_section_height)
    
    index_row = int(index/n_sections_width)
    index_col = index%n_sections_width

    return  (index_col*offset_width, index_row*offset_height,
             index_col*offset_width + target_section_width,
             index_row*offset_height + target_section_height)


def mark_prediction(map_file_path, marked_map_file_path, predictions,
                    target_section_width = 128,
                    target_section_height = 128):

    map_image = Image.open(map_file_path)
    map_draw = ImageDraw.Draw(map_image)

    n_sections = predictions.shape[0]
    n_classes = predictions.shape[1]

    for index in range(n_sections):
        box = get_box_from_index(index, map_image.width, map_image.height,
                                 target_section_width=target_section_width,
                                 target_section_height=target_section_height)
        box_center = ((box[2] + box[0])/2, (box[3] + box[1])/2)

        map_draw.text(box_center, index_to_class(np.argmax(predictions[index])))

    map_image.save(marked_map_file_path)


def index_to_class(index):
    classes = ['Airport', 'Beach', 'Bridge', 'Center', 'Church',
               'DenseResidential', 'Farmland', 'Forest', 'Industrial',
               'Meadow', 'MediumResidential', 'Mountain', 'Park', 'Parking',
               'Playground', 'Pond', 'Port', 'RailwayStation', 'River',
               'School', 'SparseResidential', 'Square', 'Stadium',
               'StorageTanks', 'Viaduct']
    
    return classes[index]

def class_to_index(neural_class):
    classes = ['Airport', 'Beach', 'Bridge', 'Center', 'Church',
               'DenseResidential', 'Farmland', 'Forest', 'Industrial',
               'Meadow', 'MediumResidential', 'Mountain', 'Park', 'Parking',
               'Playground', 'Pond', 'Port', 'RailwayStation', 'River',
               'School', 'SparseResidential', 'Square', 'Stadium',
               'StorageTanks', 'Viaduct']
    
    for i in range(len(classes)):
        if classes[i] == neural_class:
            return i


#example usage:

#divide_and_conquer('./source/source.png', './destination')
#gen_img_data = load_sections('./destination')
#pred = get_predictions(gen_img_data, './net_data/model_arh2.json', './net_data/model_w_102.h5')

#mark_prediction('./source/source.png', pred)
