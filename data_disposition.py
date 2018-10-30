import os
from shutil import copyfile


def create(train_percentage, aid_dir="/AID/"):

    root = os.getcwd()
    data_path = root + "\\Data\\"
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        train_dir = data_path + "TRAIN\\"
        os.makedirs(train_dir)
        test_dir = data_path + "TEST\\"
        os.makedirs(test_dir)
    else:
        train_dir = data_path + "TRAIN\\"
        test_dir = data_path + "TEST\\"

    for root, dirs, files in os.walk(root + aid_dir, topdown=False):
        for name in dirs:
            path = os.path.join(root, name)
            print("Transferring photos from '" + path + "' ...")

            os.chdir(path)

            file = os.listdir()
            print("Total number of images " + str(len(file)))

            total_examples = len(os.listdir())
            train_examples = int(train_percentage * total_examples)
            os.makedirs(os.path.join(train_dir, name))
            os.makedirs(os.path.join(test_dir, name))

            for i in range(train_examples):
                copyfile(os.path.join(root, name, file[i]), os.path.join(train_dir, name, file[i]))
            for i in range(train_examples, total_examples):
                copyfile(os.path.join(root, name, file[i]), os.path.join(test_dir, name, file[i]))
