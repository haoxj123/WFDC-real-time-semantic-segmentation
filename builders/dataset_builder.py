import os
import pickle
from torch.utils import data
from dataset.cityscapes import CityscapesDataSet, CityscapesTrainInform, CityscapesValDataSet, CityscapesTestDataSet


def build_dataset_train(dataset, input_size, batch_size, train_type, random_scale, random_mirror, num_workers):
    data_dir = os.path.join('./dataset/', dataset)
    dataset_list = os.path.join(dataset, '_trainval_list.txt')
    train_data_list = os.path.join(data_dir, dataset + '_' + train_type + '_list.txt')
    val_data_list = os.path.join(data_dir, dataset + '_val' + '_list.txt')
    inform_data_file = os.path.join('./dataset/inform/', dataset + '_inform.pkl')

    if not os.path.isfile(inform_data_file):
        print("%s is not found" % (inform_data_file))
        if dataset == "cityscapes":
            dataCollect = CityscapesTrainInform(data_dir, 19, train_set_file=dataset_list,
                                                inform_data_file=inform_data_file)
       

        datas = dataCollect.collectDataAndSave()
        if datas is None:
            print("error while pickling data. Please check.")
            exit(-1)
    else:
        print("find file: ", str(inform_data_file))
        datas = pickle.load(open(inform_data_file, "rb"))

    if dataset == "cityscapes":

        trainLoader = data.DataLoader(
            CityscapesDataSet(data_dir, train_data_list, crop_size=input_size, scale=random_scale,
                              mirror=random_mirror, mean=datas['mean']),
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True, drop_last=True)

        valLoader = data.DataLoader(
            CityscapesValDataSet(data_dir, val_data_list, f_scale=1, mean=datas['mean']),
            batch_size=1, shuffle=True, num_workers=num_workers, pin_memory=True,
            drop_last=True)

        return datas, trainLoader, valLoader



def build_dataset_test(dataset, num_workers, none_gt=False):
    data_dir = os.path.join('./dataset/', dataset)
    dataset_list = os.path.join(dataset, '_trainval_list.txt')
    test_data_list = os.path.join(data_dir, dataset + '_test' + '_list.txt')
    inform_data_file = os.path.join('./dataset/inform/', dataset + '_inform.pkl')

    if not os.path.isfile(inform_data_file):
        print("%s is not found" % (inform_data_file))
        if dataset == "cityscapes":
            dataCollect = CityscapesTrainInform(data_dir, 19, train_set_file=dataset_list,
                                                inform_data_file=inform_data_file)
        
        datas = dataCollect.collectDataAndSave()
        if datas is None:
            print("error while pickling data. Please check.")
            exit(-1)
    else:
        print("find file: ", str(inform_data_file))
        datas = pickle.load(open(inform_data_file, "rb"))

    if dataset == "cityscapes":
        if none_gt:
            testLoader = data.DataLoader(
                CityscapesTestDataSet(data_dir, test_data_list, mean=datas['mean']),
                batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
        else:
            test_data_list = os.path.join(data_dir, dataset + '_val' + '_list.txt')
            testLoader = data.DataLoader(
                CityscapesValDataSet(data_dir, test_data_list, mean=datas['mean']),
                batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

        return datas, testLoader

