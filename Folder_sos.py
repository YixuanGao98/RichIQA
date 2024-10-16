import torch.utils.data as data

from PIL import Image
import h5py
import os
import os.path
#import math
import scipy.io
import numpy as np
import random
import csv

def getFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    # print f_list
    for i in f_list:
        if os.path.splitext(i)[1] == suffix:
            filename.append(i)
    return filename

def getDistortionTypeFileName(path, num):
    filename = []
    index = 1
    for i in range(0,num):
        name = '%s%s%s' % ('img',str(index),'.bmp')
        filename.append(os.path.join(path,name))
        index = index + 1
    return filename
        

class CSIQFolder(data.Dataset):

    def __init__(self, root, loader, index, transform=None, target_transform=None):
        self.loader = loader

        refpath = os.path.join(root, 'src_imgs')
        refname = getFileName(refpath,'.png')



        self.histlabels = []
        self.imgname=[]
        refnames_all = []
        self.csv_file = 'bin5/CSIQhist_5.txt'
        # self.csv_file ='/home/gyx/DATA/imagehist/CSIQ/CSIQhist.txt'
        self.mos=[]
        
        with open(self.csv_file) as f:
            reader = f.readlines()
            for i, line in enumerate(reader):
                token = line.split("\t")
                token[0]=eval(token[0]) #LIVE去除字符串两端的引号
                self.imgname.append(token[0])
                values = np.array(token[1:6], dtype='float32')
                values /= values.sum()
                self.histlabels.append(values)

                ref_temp = token[0].split(".")
                refnames_all.append(ref_temp[0] + '.' + ref_temp[-1])
                mos=eval(token[6])
                self.mos.append(mos)
        

        # moslabels = np.array(target).astype(np.float32)
        refnames_all = np.array(refnames_all)

        sample = []

        for i, item in enumerate(index):
            train_sel = (refname[index[i]] == refnames_all)
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[0].tolist()
            for j, item in enumerate(train_sel):
                # for aug in range(patch_num):
                sample.append((os.path.join(root, 'dst_imgs_all', self.imgname[item]), self.histlabels[item], self.mos[item]))
        self.samples = sample    
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target,mos = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample,sample,mos, target

    def __len__(self):
        length = len(self.samples)
        return length



import pandas as pd




class KadidFolder(data.Dataset):
    def __init__(self, root, loader, index, transform=None, target_transform=None):
        self.loader = loader
        # mos_all = []
        csv_file = os.path.join(root, 'dmos.csv')
        df = pd.read_csv(csv_file)
        imgnames = df['dist_img'].tolist()
        # print(imgnames)
        labels = np.array(df['dmos']).astype(np.float32)
        # print(labels)
        refname = np.unique(np.array(df['ref_img']))
        refnames_all = np.array(df['ref_img'])
        # print(refnames_all)
        self.histlabels = []
        self.csv_file = 'bin5/kadid10khist_5.txt'
        with open(self.csv_file) as f:
            reader = f.readlines()
            for i, line in enumerate(reader):
                token = line.split("\t")
                values = np.array(token[1:6], dtype='float32')
                values /= values.sum()
                self.histlabels.append(values)

        sample = []
        for i, item in enumerate(index):
            # print(refname[index[i]])
            train_sel = (refname[index[i]] == refnames_all)
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[0].tolist()
            # print(train_sel)
            for j, item in enumerate(train_sel):
                # if sel =='all':
                # for aug in range(patch_num):
                sample.append((os.path.join(root, 'images', imgnames[item]), self.histlabels[item], labels[item]))
                # elif sel == int(imgnames[item].split('_')[1]):
                #     for aug in range(patch_num):
                #         sample.append((os.path.join(root, 'images', imgnames[item]), self.histlabels[item], labels[item]))

        self.samples = sample    
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target,mos = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample,sample,mos, target

    def __len__(self):
        length = len(self.samples)
        return length




class BIDFolder(data.Dataset):

    def __init__(self, root, loader, index, transform=None, target_transform=None):

        self.root = root
        self.loader = loader
        self.sos=[]
        self.imgname = []
        self.labels = []
        self.mos=[]
        self.cls=[]
        self.csv_file = 'bin5/BIDhist.txt'
        with open(self.csv_file) as f:
            reader = f.readlines()
            for i, line in enumerate(reader):
                token = line.split("\t")
                token[0]=eval(token[0]) #LIVE去除字符串两端的引号
                self.imgname.append(token[0])
                values = np.array(token[1:6], dtype='float32')
                values /= values.sum()
                self.labels.append(values)
                mos=eval(token[6])
                self.mos.append(mos)
                sos=eval(token[7])
                self.sos.append(sos)  
                self.cls.append(round(mos-1))
        self.cls_num_list = [np.sum(np.array(self.cls)==i) for i in range(5)]
        sample = []

        

        for j, item in enumerate(index):
            sample.append((os.path.join(self.root,'ImageDatabase', self.imgname[item]), self.labels[item], self.mos[item], self.sos[item]))
            # sample.append((self.imgpath[item],self.labels[0][item]))
        self.samples = sample    
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target,mos,sos = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample,sample,mos,sos, target

    def __len__(self):
        length = len(self.samples)
        return length





class LIVEChallengeFolder(data.Dataset):

    def __init__(self, root, loader, index, transform=None, target_transform=None):
        
        # imgpath = scipy.io.loadmat(os.path.join(root, 'Data', 'AllImages_release.mat'))
        # imgpath = imgpath['AllImages_release']
        # imgpath = imgpath[7:1169]
        # mos = scipy.io.loadmat(os.path.join(root, 'Data', 'AllMOS_release.mat'))
        # labels = mos['AllMOS_release'].astype(np.float32)
        # labels = labels[0][7:1169]

        self.root = root
        self.loader = loader
        self.cls=[]
        self.imgname = []
        self.labels = []
        self.mos=[]
        self.sos=[]
        self.csv_file = 'bin5/CLIVEhist_5.txt'
        with open(self.csv_file) as f:
            reader = f.readlines()
            for i, line in enumerate(reader):
                token = line.split("\t")
                token[0]=eval(token[0]) #LIVE去除字符串两端的引号
                self.imgname.append(token[0])
                values = np.array(token[1:6], dtype='float32')
                # values /= values.sum()
                self.labels.append(values)
                mos=eval(token[6])
                self.mos.append(mos)

                sos=eval(token[7])
                self.sos.append(sos)  
                self.cls.append(round(mos-1))
        self.cls_num_list = [np.sum(np.array(self.cls)==i) for i in range(5)]
        sample = []
        for i, item in enumerate(index):
            # for aug in range(patch_num):
            sample.append((os.path.join(self.root,'Images', self.imgname[item]), self.labels[item], self.mos[item], self.sos[item]))

        self.samples = sample    
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target,mos,sos = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample,sample,mos,sos, target

    def __len__(self):
        length = len(self.samples)
        return length




class LIVEMDFolder(data.Dataset):

    def __init__(self, root, loader, index, transform=None, target_transform=None):

        self.root = root
        self.loader = loader
        self.refpath = os.path.join(self.root, 'refimgs')
        self.refname = getFileName( self.refpath,'.bmp')

        # self.histlabels = []
        self.imgname = []
        refnames_all = []
        self.labels = []
        self.mos=[]
        self.csv_file = 'bin5/LIVEMDhist_5.txt'
        with open(self.csv_file) as f:
            reader = f.readlines()
            for i, line in enumerate(reader):
                token = line.split("\t")
                token[0]=eval(token[0]) #LIVE去除字符串两端的引号
                self.imgname.append(token[0])
                values = np.array(token[1:6], dtype='float32')
                # values /= values.sum()
                self.labels.append(values)
                mos=eval(token[6])
                self.mos.append(mos)
        refnames_all = scipy.io.loadmat(os.path.join(self.root, 'refnames_all.mat'))
        self.refnames_all = refnames_all['refnames_all']

        sample = []
        

        for i, item in enumerate(index):
            # print(refname[index[i]])
            train_sel = (self.refname[index[i]] == self.refnames_all)
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[0].tolist()
            for j, item in enumerate(train_sel):
                sample.append((os.path.join(self.root,'allimage', self.imgname[item]), self.labels[item], self.mos[item]))
            # sample.append((self.imgpath[item],self.labels[0][item]))
        self.samples = sample    
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target,mos = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample,sample,mos, target

    def __len__(self):
        length = len(self.samples)
        return length






class CID2013Folder(data.Dataset):

    def __init__(self, root, loader, index, transform=None, target_transform=None):

        self.root = root
        self.loader = loader

        self.imgname = []
        self.labels = []
        self.mos=[]
        self.csv_file = 'bin5/CID2013hist_5.txt'
        with open(self.csv_file) as f:
            reader = f.readlines()
            for i, line in enumerate(reader):
                token = line.split("\t")
                token[0]=eval(token[0]) #LIVE去除字符串两端的引号
                self.imgname.append(token[0]+'.jpg')
                values = np.array(token[1:6], dtype='float32')
                values /= values.sum()
                self.labels.append(values)
                mos=eval(token[6])
                self.mos.append(mos)
        

        sample = []
        

        for j, item in enumerate(index):
            sample.append((os.path.join(self.root,'CIDimg', self.imgname[item]), self.labels[item], self.mos[item]))
            # sample.append((self.imgpath[item],self.labels[0][item]))
        self.samples = sample    
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target,mos = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample,sample,mos, target

    def __len__(self):
        length = len(self.samples)
        return length


class SPAQFolder:
    def __init__(self, root, loader, index, transform=None, target_transform=None):


        self.root = root
        self.loader = loader

        self.cls=[]
        self.imgname = []
        self.labels = []
        self.mos=[]
        self.sos=[]
        self.csv_file = 'bin5/SPAQhist.txt'
        with open(self.csv_file) as f:
            reader = f.readlines()
            for i, line in enumerate(reader):
                token = line.split("\t")
                token[0]=eval(token[0]) #LIVE去除字符串两端的引号
                self.imgname.append(token[0])
                values = np.array(token[1:6], dtype='float32')
                values /= values.sum()
                self.labels.append(values)
                mos=eval(token[6])
                self.mos.append(mos)
                sos=eval(token[7])
                self.sos.append(sos)  
                self.cls.append(round(mos-1))
        self.cls_num_list = [np.sum(np.array(self.cls)==i) for i in range(5)]
        sample = []
        
        for i, item in enumerate(index):
            sample.append((os.path.join(self.root, 'TestImage', self.imgname[item]), self.labels[item],self.mos[item], self.sos[item]))

        self.samples = sample    
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target,mos,sos = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample,sample,mos,sos, target

    def __len__(self):
        length = len(self.samples)
        return length



class VCLFERFolder(data.Dataset):

    def __init__(self, root, loader, index, transform=None, target_transform=None):
        self.loader = loader

        refpath = os.path.join(root, 'Ref_Images')
        refname = getFileName(refpath,'.bmp')



        self.histlabels = []
        self.imgname=[]
        refnames_all = []
        self.csv_file = 'bin5/VCLFERhist.txt'
        # self.csv_file ='/home/gyx/DATA/imagehist/CSIQ/CSIQhist.txt'
        self.mos=[]
        self.std=[]
        with open(self.csv_file) as f:
            reader = f.readlines()
            for i, line in enumerate(reader):
                token = line.split("\t")
                token[0]=eval(token[0]) #LIVE去除字符串两端的引号
                token0 = token[0].split("/")
                token1=token0[2] #LIVE去除字符串两端的引号
                self.imgname.append(token1)
                values = np.array(token[1:6], dtype='float32')
                values /= values.sum()
                self.histlabels.append(values)

                ref_temp = token1.split(".")
                ref=ref_temp[0]
                refnames_all.append(ref[0:6] + '.bmp' )
                mos=eval(token[6])
                self.mos.append(mos)
                std=eval(token[7])
                self.std.append(std)

        # moslabels = np.array(target).astype(np.float32)
        refnames_all = np.array(refnames_all)

        sample = []

        for i, item in enumerate(index):
            train_sel = (refname[index[i]] == refnames_all)
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[0].tolist()
            for j, item in enumerate(train_sel):
                # for aug in range(patch_num):
                sample.append((os.path.join(root, 'vcl_fer', self.imgname[item]), self.histlabels[item], self.mos[item]))
        self.samples = sample    
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target,mos = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample,sample,mos, target

    def __len__(self):
        length = len(self.samples)
        return length


class FLIVEFolder:
    def __init__(self, root, loader, index, transform=None, target_transform=None):


        self.root = root
        self.loader = loader
        self.cls=[]
        self.imgname = []
        self.labels = []
        self.mos=[]
        self.sos=[]
        self.csv_file = 'bin5/FLIVEhist_5.txt'
        with open(self.csv_file) as f:
            reader = f.readlines()
            for i, line in enumerate(reader):
                token = line.split("\t")
                # token[0]=eval(token[0]) #LIVE去除字符串两端的引号
                # a=token[0][:-1]
                self.imgname.append(token[0][:-1])
                values = np.array(token[1:6], dtype='float32')
                values /= values.sum()
                self.labels.append(values)
                mos=eval(token[6])
                self.mos.append(mos)
                sos=eval(token[7])
                self.sos.append(sos)  
                self.cls.append(round(mos-1))
        self.cls_num_list = [np.sum(np.array(self.cls)==i) for i in range(5)]
        sample = []
        
        
        for i, item in enumerate(index):
            sample.append((os.path.join(self.root, self.imgname[item]), self.labels[item], self.mos[item], self.sos[item]))

        self.samples = sample    
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target,mos,sos = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample,sample,mos,sos, target

    def __len__(self):
        length = len(self.samples)
        return length

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

if __name__ == '__main__':
    liveroot = '/home/gyx/DATA/imagehist/CSIQ'
    index = list(range(0,30))
    random.shuffle(index)
    train_index = index[0:round(0.8*30)]
    test_index = index[round(0.8*30):30]
    trainset = CSIQFolder(root = liveroot, loader = default_loader, index = train_index)
    testset = CSIQFolder(root = liveroot, loader = default_loader, index = test_index)
