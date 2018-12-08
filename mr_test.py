import map_recognition as mr
import numpy as np


#mr.divide_and_conquer('./source/source.png', './destination')
gen_img_data = mr.load_sections('./destination')
#gen_img_data = mr.load_sections('./testground')
pred = mr.get_predictions(gen_img_data, './model_arh2.json', './model_w_102.h5')

#print(mr.index_to_class(np.argmax(pred[0])))


#total = 0
#for i in range(pred.shape[0]):
#    if pred[i][mr.class_to_index('Airport')] > 0.5:
#        total = total + 1
#print(total)

mr.mark_prediction('./source/source.png', './source/marked.png', pred)