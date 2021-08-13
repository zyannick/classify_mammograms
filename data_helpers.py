import numpy as np
import cv2
import json
from glob import glob
import os
import random
import tensorflow as tf
from PIL import Image
import utils



def augmentation(batch, random):
    if random[0] < 0.3:
        batch_flip = np.flip(batch, 1)
    elif random[0] > 0.7:
        batch_flip = np.flip(batch, 2)
    else:
        batch_flip = batch

    if random[1] < 0.5:
        batch_rot = np.rot90(batch_flip, 1, [1, 2])
    else:
        batch_rot = batch_flip

    return batch_rot

def get_batch(data_root, batch_size, dataset_name, list_files, image_size, channel, num_classes, index = None):

    data_num = len(list_files)

    if index is not None and  batch_size !=1:
        raise  ValueError('Error on the index and the batch size')

    if index is None:
        random_batch = np.random.rand(batch_size) * (data_num - 1)

    batch_images = np.zeros([batch_size, image_size, image_size, channel])
    batch_labels = np.zeros([batch_size, num_classes])

    list_images = []
    
    for i in range(batch_size):
        if index is None:
            ri = int(random_batch[i])
        else:
            ri = index
        patient_id, record_path, filename, str_label =  list_files[ri]
        file_path = os.path.join(data_root, dataset_name,record_path, filename)

        list_images.append(file_path)

        # img = Image.open(file_path)
        # img = img.resize((image_size,image_size), Image.ANTIALIAS)
        img = cv2.imread(file_path, 0)
        img = cv2.resize(img, dsize=(image_size, image_size))
        img = np.asfarray(img)
        img = img / 255.
        img = np.expand_dims(img, axis= 2)

        label = [1, 0] if str_label == 'MLO' else [0, 1]

        batch_images[i, :, :, :] = np.asarray(img)
        batch_labels[i, :] = np.asarray(label)


    # random_aug = np.random.rand(2)
    # batch_images = augmentation(batch_images, random_aug)

    
    return batch_images, batch_labels, list_images

def get_image(data_root, dataset_name, list_files, image_size, channel, num_classes):

    batch_size = 1

    data_num = len(list_files)

    random_batch = np.random.rand(batch_size) * (data_num - 1)
    batch_images = np.zeros([batch_size, image_size, image_size, channel])
    batch_labels = np.zeros([batch_size, num_classes])
    
    for i in range(batch_size):
        ri = int(random_batch[i])
        patient_id, record_path, filename, str_label =  list_files[ri]
        file_path = os.path.join(data_root, dataset_name,record_path, filename)

        img = Image.open(file_path)
        img = img.resize((image_size,image_size), Image.ANTIALIAS)
        img = np.asfarray(img)
        img = img / 255.
        img = np.expand_dims(img, axis= 2)

        label = [1, 0] if str_label == 'MLO' else [0, 1]

        batch_images[i, :, :, :] = np.asarray(img)
        batch_labels[i, :] = np.asarray(label)


    random_aug = np.random.rand(2)
    batch_images = augmentation(batch_images, random_aug)

    
    return batch_images, batch_labels



def get_data(flags):

    # Create a dictionnary of ally patients informations
    patient_dict = {}

    # Get the data containing files with the labels and all others informations
    data_json = os.path.join(flags.data_root, flags.data_json)
    # Check if the file exists
    if not os.path.exists(data_json):
        raise FileNotFoundError('The file data json file  {} does not exist. The program will exit now.'.format(data_json))

    with open(data_json) as json_file:
        original_dict = json.load(json_file)

    # Check if the files are already created 
    train_test_dicts = glob(os.path.join(flags.data_root, 'generated_files', '*.json'))

    data_lists = {
        'train': [],
        'test': []
    }

    all_data_dict = {
        'train': {},
        'test': {}
    }
        

    # The files are not created yet. Let's create them
    if len(train_test_dicts) < 2:

        utils.mkdirs(os.path.join(flags.data_root, 'generated_files'))

        for start_name in original_dict.keys():
            patient_id = original_dict[start_name]['patientId']
            if patient_id not in patient_dict.keys():
                patient_dict[patient_id] = {}
            record_path = original_dict[start_name]['accessionNumber']
            filename = start_name +  '.dcm.png'
            label = original_dict[start_name]['view']

            patient_dict[patient_id][start_name] = {}
            patient_dict[patient_id][start_name]['record_path'] = record_path
            patient_dict[patient_id][start_name]['filename'] = filename
            patient_dict[patient_id][start_name]['label'] = label

        

        list_patients_id = list(patient_dict.keys())
        l = len(list_patients_id)
        random.shuffle(list_patients_id)
        # Partition patient between training and testing groups
        partitionned_lists = {
            'train': list_patients_id[0:int(0.8*l)], 
            'test': list_patients_id[int(0.8*l): -1]
        }


        for phase in partitionned_lists.keys():
            list_patients = partitionned_lists[phase]
            for patient_id in list_patients:
                for start_name in patient_dict[patient_id].keys():
                    data_lists[phase].append(
                        (patient_id, patient_dict[patient_id][start_name]['record_path'], patient_dict[patient_id][start_name]['filename'], patient_dict[patient_id][start_name]['label'])
                    )
                    all_data_dict[phase][patient_id] = patient_dict[patient_id]

        with open(os.path.join(flags.data_root, 'generated_files', 'all_data.json'), 'w') as fp:
            json.dump(all_data_dict, fp,  indent=4)


        for phase in all_data_dict.keys():
            # Random suffling of the data
            random.shuffle(data_lists[phase])
            random.shuffle(data_lists[phase])
            # Save json file for after
            with open(os.path.join(flags.data_root, 'generated_files',  phase+'_dict.json'), 'w') as fp:
                json.dump(all_data_dict[phase], fp,  indent=4)
    else:
        for phase in data_lists.keys():
            with open(os.path.join(flags.data_root, 'generated_files', phase+'_dict.json')) as json_file:
                all_data_dict[phase] = json.load(json_file)
            for patient_id in all_data_dict[phase].keys():
                for start_name in all_data_dict[phase][patient_id].keys():
                    data_lists[phase].append(
                        (patient_id, all_data_dict[phase][patient_id][start_name]['record_path'], all_data_dict[phase][patient_id][start_name]['filename'], all_data_dict[phase][patient_id][start_name]['label'])
                    )

    return data_lists['train'], data_lists['test']