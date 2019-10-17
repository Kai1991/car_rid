import os

from utils import generator_batch_triplet,generator_batch_multitask

def filter_data_list(data_list):
    # data_list  : a list of [img_path, vehicleID, modelID, colorID]
    # {modelID: {colorID: {vehicleID: [imageName, ...]}}, ...}
    # dic helps us to sample positive and negative samples for each anchor.
    # https://arxiv.org/abs/1708.02386
    # The original paper says that "only the hardest triplets in which the three images have exactly
    # the same coarse-level attributes (e.g. color and model), can be used for similarity learning."
    dic = { }
    # We construct a new data list so that we could sample enough positives and negatives.
    new_data_list = [ ]
    for line in data_list:
        imgPath, vehicleID, modelID, colorID = line.strip().split(' ')
        dic.setdefault(modelID, { })
        dic[modelID].setdefault(colorID, { })
        dic[modelID][colorID].setdefault(vehicleID, [ ]).append(imgPath)

    # https://stackoverflow.com/questions/11277432/how-to-remove-a-key-from-a-python-dictionary
    for line in data_list:
        imgPath, vehicleID, modelID, colorID = line.strip().split(' ')
        #print(imgPath, vehicleID, modelID, colorID)
        if modelID in dic and colorID in dic[modelID] and vehicleID in dic[modelID][colorID] and \
                                                      len(dic[modelID][colorID][vehicleID]) == 1:
            dic[modelID][colorID].pop(vehicleID, None)

    for line in data_list:
        imgPath, vehicleID, modelID, colorID = line.strip().split(' ')
        if modelID in dic and colorID in dic[modelID] and len(dic[modelID][colorID].keys()) == 1:
            dic[modelID].pop(colorID, None)

    for modelID in dic:
        for colorID in dic[modelID]:
            for vehicleID in dic[modelID][colorID]:
                for imgPath in dic[modelID][colorID][vehicleID]:
                    new_data_list.append('{} {} {} {}'.format(imgPath, vehicleID, modelID, colorID))

    print('The original data list has {} samples, the new data list has {} samples.'.format(
                                 len(data_list), len(new_data_list)))
    return new_data_list, dic


def gen_data_train_wrap(path,config,mode):
    data_lines = open(path).readlines()
    # Check if image path exists.
    data_lines = [w for w in data_lines if os.path.exists(w.strip().split(' ')[0])]
    data_lines, dic_data_lines = filter_data_list(data_lines)
    nbr_train = len(data_lines)
    print('# Train Images: {}.'.format(nbr_train))
    if mode == "branch_color":
        return generator_batch_multitask(data_lines,
                        nbr_class_one = config.NBR_MODELS, nbr_class_two = config.NBR_COLORS,
                        batch_size = config.BATCH_SIZE, img_width = config.IMG_WIDTH,
                        img_height = config.IMG_HEIGHT, random_scale = config.RANDOM_SCALE,
                        shuffle = True, augment = True)
    elif mode == "branch_color_triplet":
        return generator_batch_triplet(data_lines, dic_data_lines,
                        mode = 'train', nbr_class_one = config.NBR_MODELS, nbr_class_two = config.NBR_COLORS,
                        batch_size = config.BATCH_SIZE, img_width = config.IMG_WIDTH,
                        img_height = config.IMG_HEIGHT, random_scale = config.RANDOM_SCALE,
                        shuffle = True, augment = True)

def gen_data_val_wrap(path,config,mode):
    data_lines = open(path).readlines()
    # Check if image path exists.
    data_lines = [w for w in data_lines if os.path.exists(w.strip().split(' ')[0])]
    data_lines, dic_data_lines = filter_data_list(data_lines)
    nbr_train = len(data_lines)
    print('# Train Images: {}.'.format(nbr_train))
    if mode == "branch_color":
        return generator_batch_multitask(data_lines,
                        nbr_class_one = config.NBR_MODELS, nbr_class_two = config.NBR_COLORS,
                        batch_size = config.BATCH_SIZE, img_width = config.IMG_WIDTH,
                        img_height = config.IMG_HEIGHT, random_scale = config.RANDOM_SCALE,
                        shuffle = True, augment = True)
    elif mode == "branch_color_triplet":
        return generator_batch_triplet(data_lines, dic_data_lines,
                        mode = 'val', nbr_class_one = config.NBR_MODELS, nbr_class_two = config.NBR_COLORS,
                        batch_size = config.BATCH_SIZE, img_width = config.IMG_WIDTH,
                        img_height = config.IMG_HEIGHT, random_scale = config.RANDOM_SCALE,
                        shuffle = True, augment = False)

