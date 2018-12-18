import map_recognition as mr
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from PIL import Image
import pickle


map_image_path = './source/source.png'

map_img = Image.open(map_image_path)

n_sec_w, n_sec_h = mr.n_sec(map_img)

num_sections = n_sec_w*n_sec_h
#num_sections = 30

#num_sections = mr.divide_and_conquer('./source/source.png', './destination')
#num_sections = mr.divide_and_conquer2(map_image_path, './destination')
#gen_img_data = mr.load_sections('./destination')
#gen_img_data = mr.load_sections('./testground')

model = mr.load_model('./model/test2-L2/model_arh2.json', './model/test2-L2/weights00000016.h5')
generator = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
#pred = mr.get_predictions(gen_img_data, model)
#print(mr.index_to_class(np.argmax(pred[0])))

'''
img = mr.load_img_to_array('source\\2.png')
pred = mr.predict_img(model, img, generator)

sorted_pred = mr.sort_class_conf(pred)
for i in sorted_pred:
    print(i)
'''

'''
predictions = np.zeros((num_sections, mr.get_classes_num()))
for i in range(num_sections):
    print('Working on ' + str(i+1) + ' from ' + str(num_sections))
    img = mr.load_img_to_array('destination\\unclassified\\' + str(i) + '.png')
    predictions[i] += mr.predict_img(model, img, generator)[0]
'''
'''
for i in range(1,6):
    img_box = mr.scaled_section_box(map_img, 406, i)
    print(i, img_box, img_box[0]-img_box[2], img_box[1]-img_box[3])
'''

predictions = np.zeros((num_sections, mr.get_classes_num()))
for i in range(num_sections):
    print('Working on ' + str(i+1) + ' from ' + str(num_sections))
    img_box = mr.index_to_box(i, map_img)
    img = map_img.crop(img_box)
    img = mr.resize_img_to_array(img)
    #img = mr.load_img_to_array('destination\\unclassified\\' + str(i) + '.png')
    predictions[i] += mr.predict_img(model, img, generator)[0]


pred_scales = np.zeros((num_sections, 1))

for i in range(num_sections):
    print('Working on ' + str(i+1) + ' from ' + str(num_sections))
    max_conf_class = np.argmax(predictions[i])
    max_conf = predictions[i][max_conf_class]
    if max_conf > 0.7 and max_conf < 0.92:
        for j in range(1, 6):
            scaled_section_box = mr.scaled_section_box(map_img, i, j)
            scaled_section = map_img.crop(scaled_section_box)
            scaled_section = mr.resize_img_to_array(scaled_section)
            scal_pred = mr.predict_img(model, scaled_section, generator)[0]
            if np.amax(scal_pred) > max_conf:
                max_conf_class = np.argmax(scal_pred)
                max_conf = scal_pred[max_conf_class]
                np.copyto(predictions[i], scal_pred)
                pred_scales[i][0] = j

print(pred_scales)

with open('pickled/predictions.pickle', 'wb') as handle:
    pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('pickled/pred_scales.pickle', 'wb') as handle:
    pickle.dump(pred_scales, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('pickled/predictions.pickle', 'rb') as handle:
    predictions = pickle.load(handle)
with open('pickled/pred_scales.pickle', 'rb') as handle:
    pred_scales = pickle.load(handle)

#total = 0
#for i in range(pred.shape[0]):
#    if pred[i][mr.class_to_index('Airport')] > 0.5:
#        total = total + 1
#print(total)

#mr.mark_prediction('./source/source.png', './source/marked.png', pred)
#mr.mark_prediction2('./source/source.png', './source/marked.png', predictions)
mr.mark_prediction3('./source/source.png', './source/marked.png', predictions, pred_scales)