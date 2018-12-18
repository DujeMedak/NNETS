from keras_preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import model_helper
from PIL import Image, ImageDraw, ImageFont
import os
import math
from random import random


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

    return n_sections_height*n_sections_width

def divide_and_conquer2(map_file_path, output_map_sections_path,
                       target_section_width = 128,
                       target_section_height = 128,
                       class_subfolder = 'unclassified',
                       sections_file_format = 'png'):

    whole_map = Image.open(map_file_path)

    output_path = output_map_sections_path + '/' + class_subfolder + '/'

    # Create output directory if it doesnt exitst.
    os.makedirs(output_path, exist_ok = True)

    i = 0
    box = index_to_box(i, whole_map)
    while box:
        region = whole_map.crop(box)
        region.save(output_path + str(i) + '.' + sections_file_format)
        
        i += 1
        box = index_to_box(i, whole_map)

    return i



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

def index_to_box(index, img,
                 overlap = 0.5,
                 target_section_width = 128,
                 target_section_height = 128):

    n_sec_w, n_sec_h = n_sec(img)

    if index >= n_sec_w*n_sec_h:
        return None

    x = (index%n_sec_w)*int((1-overlap)*target_section_width)
    y = int(index/n_sec_w)*int((1-overlap)*target_section_height)

    return (x, y, x + target_section_width, y + target_section_height)

def n_sec(img,
          overlap = 0.5,
          target_section_width = 128,
          target_section_height = 128):

    img_w = img.width
    img_h = img.height

    n_sec_w = int((img_w - int(overlap*target_section_width))/((1-overlap)*target_section_width))
    n_sec_h = int((img_h - int(overlap*target_section_height))/((1-overlap)*target_section_height))

    return n_sec_w, n_sec_h



def load_img_to_array(path,
                      target_section_width = 128,
                      target_section_height = 128):

    img = load_img(path)  # this is a PIL image
    return resize_img_to_array(img, target_section_width = target_section_width,
                                    target_section_height = target_section_height)


def resize_img_to_array(img,
                        target_section_width = 128,
                        target_section_height = 128):

    img.thumbnail((target_section_width, target_section_height), Image.ANTIALIAS)
    x = img_to_array(img)  # this is a Numpy array with shape (3, W, H)
    return x.reshape((1,) + x.shape)

def scaled_section_box(map_img, sec_index, scale_level,
                       scale_increment = 0.2,
                       base_section_width = 128,
                       base_section_height = 128):

    base_box = index_to_box(sec_index, map_img)

    scale_inc_w = int(scale_level*scale_increment*base_section_width)
    scale_inc_h = int(scale_level*scale_increment*base_section_height)

    if (scale_inc_w + base_section_width) >= map_img.width or                  \
       (scale_inc_h + base_section_height) >= map_img.height:

        return base_box

    scaled_box = [0, 0, 0, 0]
    scaled_box[0] = base_box[0] - int(scale_inc_w/2)
    scaled_box[1] = base_box[1] - int(scale_inc_h/2)
    scaled_box[2] = base_box[2] + int(scale_inc_w/2)
    scaled_box[3] = base_box[3] + int(scale_inc_h/2)

    # Correct integer division error.
    if scale_inc_w%2:
        scaled_box[2] += 1
    if scale_inc_h%2:
        scaled_box[3] += 1

    # If box goes outside of image bounds move it back in.
    if scaled_box[0] < 0:
        scaled_box[2] = scaled_box[2] - scaled_box[0]
        scaled_box[0] = 0
    if scaled_box[1] < 0:
        scaled_box[3] = scaled_box[3] - scaled_box[1]
        scaled_box[1] = 0
    if scaled_box[2] > map_img.width:
        scaled_box[0] = scaled_box[0] + (map_img.width - scaled_box[2])
        scaled_box[2] = map_img.width
    if scaled_box[3] > map_img.height:
        scaled_box[1] = scaled_box[1] + (map_img.height - scaled_box[3])
        scaled_box[3] = map_img.height

    return tuple(scaled_box)


##########################NET PREDICTION########################################

def get_predictions(img_data, model):

    print(img_data.samples)

    # Get the predictions from the model using the generator
    predictions = model.predict_generator(img_data,
                                        steps=img_data.samples / img_data.batch_size, verbose=1)

    return predictions

def load_model(model_file_path, weights_file_path):

    model = model_helper.load_model_from_file(model_file_path)
    model.load_weights(weights_file_path)

    return model


def predict_img(model, image, generator,
                  num_img_aug=8):

    pred_sum = np.zeros((1, get_classes_num()))
    for i in range(num_img_aug):
        tmp_data = generator.flow(image, batch_size=1)
        pred_sum[0] += model.predict_generator(tmp_data, steps = 1)[0]

    '''
    pred_sum = np.zeros((1, get_classes_num()))
    for i in range(num_img_aug):
        pred_sum[0] += pred[i]
    '''
    return pred_sum/num_img_aug

    '''
    sum_sorted = sort_class_conf(pred_sum)
    for i in range(get_classes_num()):
        print(i+1, '. ', index_to_class(sum_sorted[0][i]), ': ', pred_sum[0][sum_sorted[0][i]])
    '''

def sort_class_conf(prediction):

    ranking = []
    for index in list(np.flip(np.argsort(prediction), 1)[0]):
        ranking.append({'class': index_to_class(index),
                        'confidence': prediction[0][index]})
    
    return ranking



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

        #map_draw.text(box_center, index_to_class(np.argmax(predictions[index])))
        map_draw.text(box_center, str(predictions[index][np.argmax(predictions[index])])[0:4] + ' ' + index_to_class(np.argmax(predictions[index])))

    map_image.save(marked_map_file_path)

def mark_prediction2(map_file_path, marked_map_file_path, predictions,
                     target_section_width = 128,
                     target_section_height = 128):

    map_image = Image.open(map_file_path)
    map_draw = ImageDraw.Draw(map_image)

    n_sections = predictions.shape[0]
    n_classes = predictions.shape[1]


    i = 0
    box = index_to_box(i, map_image)
    while box:
        i += 1
        box = index_to_box(i, map_image)

    for index in range(i):
        box = index_to_box(index, map_image,
                           target_section_width=target_section_width,
                           target_section_height=target_section_height)
        box_center = ((box[2] + box[0])/2, (box[3] + box[1])/2)

        if predictions[index][np.argmax(predictions[index])] > 0.8:
            #map_draw.text(box_center, index_to_class(np.argmax(predictions[index])))
            map_draw.text(box_center, str(predictions[index][np.argmax(predictions[index])])[0:4] + ' ' + index_to_class(np.argmax(predictions[index])))

    map_image.save(marked_map_file_path)

def mark_prediction3(map_file_path, marked_map_file_path, predictions, scales,
                     target_section_width = 128,
                     target_section_height = 128):

    map_image = Image.open(map_file_path)
    map_draw = ImageDraw.Draw(map_image)

    n_sections = predictions.shape[0]
    n_classes = predictions.shape[1]

    for index in range(n_sections):
        box = scaled_section_box(map_image, index, scales[index][0],
                                 base_section_width=target_section_width,
                                 base_section_height=target_section_height)
        box_left_align = (box[0], (box[3] + box[1])/2)

        if np.amax(predictions[index]) > 0.8:
            map_draw.rectangle(box, outline = (int(random()*255), int(random()*255), int(random()*255)))
            map_draw.text(box_left_align, str(np.amax(predictions[index]))[0:4] + ' ' + index_to_class(np.argmax(predictions[index])))

    map_image.save(marked_map_file_path)


classes = ['Airport', 'Beach', 'Bridge', 'Center', 'Church',
            'DenseResidential', 'Farmland', 'Forest', 'Industrial',
            'Meadow', 'MediumResidential', 'Mountain', 'Park', 'Parking',
            'Playground', 'Pond', 'Port', 'RailwayStation', 'River',
            'School', 'SparseResidential', 'Square', 'Stadium',
            'StorageTanks', 'Viaduct']

def get_classes_num():

    return len(classes)

def index_to_class(index):

    return classes[index]

def class_to_index(neural_class):
    
    for i in range(len(classes)):
        if classes[i] == neural_class:
            return i


#example usage:

#divide_and_conquer('./source/source.png', './destination')
#gen_img_data = load_sections('./destination')
#pred = get_predictions(gen_img_data, './net_data/model_arh2.json', './net_data/model_w_102.h5')

#mark_prediction('./source/source.png', pred)
