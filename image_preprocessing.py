from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array,array_to_img
import sys, os
import datetime

print ("[INFO] program started on - " + str(datetime.datetime.now()))

# how many new pictures will be created from each existing
number_of_iterations = 20 #control variable for counting of printing into folder
batch_size=32 #size of augmented picture created in 1 iteration

# create generator for augmentation, moze se stvoriti proizvoljan broj generatora s proizvoljnim funkcijama
datagen = ImageDataGenerator(
		zca_whitening=True,
		rotation_range=150,
		horizontal_flip=True,
		vertical_flip=True,
		)

# input pictures must be in the same folder as script and folder is named AID
# get the input and output path
script_dir = sys.path[0]
input_path = os.path.join(script_dir, 'AID/')
input_path = os.path.normpath(input_path)
print("source pictures: "+input_path)

os.system("mkdir " + "AID_augmented")
output_path = os.path.join(script_dir, 'AID_augmented/')
output_path = os.path.normpath(output_path)
print("augmented pictures: "+output_path)

# get the class label limit
class_limit = 24

# AID class names (aerial image dataset)
class_names = ["Airport", "Beach", "Bridge", "Center", "Church",
			   "DenseResidential", "Farmland", "Forest", "Industrial", "Meadow",
			   "MediumResidential", "Mountain", "Park", "Parking", "Playground",
			   "Pond", "Port","RailwayStation","River","SparseResidential","Square",
				"Stadium","StorageTanks","Viaduct"]
####################
# this is a generator that will read pictures found in
# subfolers of 'input_path', and indefinitely generate
# batches of augmented image data (batch size iz given
# as batch_size, and number of batches is given as number_of_iterations

# this is an example with saving generated augmented photos, for simple generator remove
# the part: for batch in and simply put generator = datagen-flow_from_directory.... and use generator
# later on for training and testing as a indefinite flow of augmented data, save_to_dir part should also be removed
# -- see below for flow example
# i = 0
# for batch in datagen.flow_from_directory(
#         input_path,  # this is the target directory
#         target_size=(299, 299),  # all images will be resized to 299x299
#         batch_size=batch_size,
# 		shuffle=False,
# 		class_mode='categorical',
# 		save_to_dir=output_path, save_prefix='test', save_format='jpg'):
# 	i += 1
# 	if i >= number_of_iterations:
# 		break
####################
# generator_podataka = datagen.flow_from_directory(
#         input_path,  # this is the target directory
#         target_size=(299, 299),  # all images will be resized to 299x299
#         batch_size=batch_size,
# 		shuffle=False,
# 		class_mode='categorical')

####################
# this part reads photo by photo in each subfoled and creates
# batch_size number of augmented photos from single picture
# for number_of_iterations times, new folder is created with name:
# AID_augmented and all following subfolders with generated augmented photos
# no datagenerator exists, you must manually create new data flow from newly created
# folder and subfolders
# # change the current working directory
# os.chdir(output_path)
# # loop over the class labels
# for x in range(0, class_limit):
# 	# create a folder for that class
# 	os.system("mkdir " + class_names[x])
# 	# take all the images from the dataset
# 	image_paths = glob.glob(input_path + "\\" + class_names[x] +"\\*.jpg")
# 	# get the current path
# 	cur_path = output_path + "\\" + class_names[x] + "\\"
#
# #TEST ZA PROVJERU AUGMENTIRANIH SLIKA
# 	# loop over the images in the dataset
# 	for image_path in image_paths:
# 		original_path = image_path
# 		image_path = image_path.split("\\")
# 		image_path = image_path[len(image_path)-1]
#
# 		single_image = load_img(original_path)
# 		single_image = img_to_array(single_image)
# 		single_image = single_image.reshape((1,) + single_image.shape)
#
# 		i = 0
# 		# with saving augmented images to file
# 		for batch in datagen.flow(single_image, batch_size=1,
# 								  save_to_dir=cur_path, save_prefix=class_names[x], save_format='jpg'):
# 			i += 1
# 			if i > number_of_augmentations:
# 				break
#

# print end time
print ("[INFO] program ended on - " + str(datetime.datetime.now()))
