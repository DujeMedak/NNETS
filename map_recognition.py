from keras_preprocessing.image import ImageDataGenerator
from PIL import Image
import os
import math


def divide_and_conquer(map_file_path, output_map_sections_path,
                       target_section_width = 128,
                       target_section_height = 128,
                       test_batchsize = 1,
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


    tf_image_data_gen = ImageDataGenerator()

    tf_image_data = tf_image_data_gen.flow_from_directory(
        output_map_sections_path,
        target_size = (target_section_height, target_section_width),
        class_mode = None,
        batch_size = 1,
        shuffle = False
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



#example usage:

#divide_and_conquer('./source/source.png', './destination')