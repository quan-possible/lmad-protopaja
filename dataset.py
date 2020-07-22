import os
import glob
import cv2
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from collections import namedtuple


#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for your approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]


class BaseDataset(Dataset):
    def __init__(self,
                 new_size=None,
                 crop_size=None,
                 random_resize=False,
                 random_color=False,
                 is_flip=False,
                 to_tensor=True,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):

        self.new_size = new_size
        self.crop_size = crop_size
        self.random_resize = random_resize
        self.random_color = random_color
        self.is_flip = is_flip
        self.to_tensor = to_tensor
        self.mean = mean
        self.std = std

        self.files = []

    def __len__(self):
        return len(self.files)

    def image_transform(self, image):
        """ Convert image's datatype to np.float32 and normalize it.

            Args:
                image (2d-array like): Input image for transformation.

            Return:
                image (2d-array like): Transformed image.
        """
        image = image.astype(np.float32)[:, :, ::-1]
        image = image / 255.0
        image -= self.mean
        image /= self.std
        return image

    def label_transform(self, label):
        """ Convert label's datatype to np.int64

            Args:
                label (2d-array like): Input label for transformation.

            Return:
                label (2d-array like): Transformed label.
        """
        return np.array(label).astype('int64')

    def pad_image(self, image, h, w, size, padvalue):
        """ Pad image with a pad value to desired size.

            Args:
                image (2d-array like): Input image for padding.
                h (int):
                w (int):
                size (tuple):
                padvalue (int): Value used for padding.

            Return:
                pad_image (2d-array like): Padded image.
        """
        pad_image = image.copy()
        pad_h = max(size[0] - h, 0)
        pad_w = max(size[1] - w, 0)
        if pad_h > 0 or pad_w > 0:
            pad_image = cv2.copyMakeBorder(image, 0, pad_h, 0,
                pad_w, cv2.BORDER_CONSTANT,
                value=padvalue)
        return pad_image

    def rand_resize(self, image, label):
        """ Randomly resize the image & label.
            The new size should not be smaller than the cropsize.

            Args:
                image (2d-array like): Image.
                label (2d-array like): Label.

            Return:
                image, label (tuple of 2d-array like):
                Random resized tuple of (image, label)
        """
        new_w = random.randint(self.crop_size[1]+1, self.new_size[0]+1)
        new_h = int(self.new_size[1] * new_w / self.new_size[0])+1
        image = cv2.resize(image, (new_w, new_h),
                           interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, (new_w, new_h),
                           interpolation = cv2.INTER_NEAREST)
        return image, label

    def rand_crop(self, image, label):
        """ Random crop image & label with the specified crop size (crop_size).

            Args:
                image (2d-array like): Image.
                label (2d-array like): Label.

            Return:
                image, label (tuple of 2d-array like):
                Random cropped tuple of (image, label)
        """
        new_h, new_w = label.shape
        x = random.randint(0, new_w - self.crop_size[1])
        y = random.randint(0, new_h - self.crop_size[0])
        image = image[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        label = label[y:y+self.crop_size[0], x:x+self.crop_size[1]]
        return image, label

    def rand_flip(self, image, label):
        """ Random flip image & label.

            Args:
                image (2d-array like): Image.
                label (2d-array like): Label.

            Return:
                image, label (tuple of 2d-array like):
                Random flipped tuple of (image, label)
        """
        flip = np.random.choice(2) * 2 - 1
        image = image[:, ::flip]
        label = label[:, ::flip]
        return image, label
    
    def rand_color(self, image, label):
        if np.random.randint(2):
            image = image + np.random.randint(0, 255, dtype="ubyte")
        return image, label

    def resize(self, image, label=None):
        """ Resize the image & label to a specified size (new_size).

            Args:
                image (2d-array like): Image.
                label (2d-array like): Label.

            Return:
                image, label (tuple of 2d-array like):
                Resized tuple of (image, label)
        """
        image = cv2.resize(image, self.new_size,
                           interpolation = cv2.INTER_LINEAR)
        if label is None:
            return image

        label = cv2.resize(label, self.new_size,
                           interpolation = cv2.INTER_NEAREST)
        return image, label

    def gen_sample(self, image, label):
        """ Generate data with specified procedure.

            Args:
                image (2d-array like): Image.
                label (2d-array like): Label.

            Return:
                image, label (tuple of 2d-array like):
                Generated tuple of (image, label)
        """
        # Random convert to grayscale:
        if self.random_color:
            image, label = self.rand_color(image, label)
        # Resize:
        if self.new_size:
            image, label = self.resize(image, label)
        # Random resize:
        if self.random_resize:
            image, label = self.rand_resize(image, label)
        # Random_crop:
        if self.crop_size:
            image, label = self.rand_crop(image, label)
        # Flip:
        if self.is_flip:
            image, label = self.rand_flip(image, label)
        # Transform:
        image = self.image_transform(image).transpose((2, 0, 1))
        label = self.label_transform(label)
        # To tensor:
        if self.to_tensor:
            image, label = torch.from_numpy(image), torch.from_numpy(label)
        # Done:
        return image, label

class Drivable(BaseDataset):
    def __init__(self,
                 root=None,
                 mode=None,
                 new_size=None,
                 crop_size=None,
                 random_resize=False,
                 random_color=False,
                 is_flip=False,
                 to_tensor=True,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):

        super(Drivable, self).__init__(new_size, crop_size, random_resize,
                                       random_color, is_flip, to_tensor, mean, std)
        self.root = root
        self.mode = mode
        # Color convert mapping:
        self.id2color = {
            0: (0, 0, 0),
            1: (255, 0, 0),
            2: (0, 255, 255)
        }
        # Read file paths:
        if root and mode:
            self.files = self.read_files()

    def read_files(self):
        """ Read and prepare file paths.

            Args:

            Return:
                files (list of dictionary of image file path & label file path)
        """
        files = []
        if 'test' in self.mode:
            imgs = sorted(glob.glob(os.path.join(self.root, 'images', self.mode, '*')))
            for img_path in imgs:
                files.append({
                    'images': img_path
                })
        else:
            imgs = sorted(glob.glob(os.path.join(self.root, 'images', self.mode, '*')))
            labs = sorted(glob.glob(os.path.join(self.root, 'labels', self.mode, '*')))
            for (img_path, lab_path) in zip(imgs, labs):
                files.append({
                    'image': img_path,
                    'label': lab_path
                })
        return files

    def convert_color(self, input, inverse=False):
        """ Original trainId <=> Color Convertion.
            Labels have to be in original trainId.
            The colorscheme follows CityScapes convention.

            Args:
                input (2d-array like): 2d array of trainId / RGB color.
                inverse (bool): If True, convert from RGB color to original
                        trainId. If False, convert original trainId to 
                        custom trainId.

            Return:
                output (2d image): Final convertion.
        """
        if inverse:
            output = np.zeros(input.shape[:2], dtype=np.uint8)
            for v, k in self.id2color.items():
                output[(input == k).sum(-1) == 3] = v
            return output
        else:
            output = np.zeros((*input.shape, 3), dtype=np.uint8)
            for k, v in self.id2color.items():
                output[input == k] = v
            return output

    def __getitem__(self, index):
        """ Get item with index.

            Return:
                image, label (tuple of 2d-array like)
        """
        item = self.files[index]
        image = cv2.imread(item["image"], cv2.IMREAD_COLOR)
        label = cv2.imread(item["label"], cv2.IMREAD_GRAYSCALE)
        image, label = self.gen_sample(image, label)
        return image, label
    
class BCG(BaseDataset):
    def __init__(self,
                 root=None,
                 mode=None,
                 classes=None,
                 new_size=None,
                 crop_size=None,
                 random_resize=False,
                 is_flip=False,
                 to_tensor=True,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):

        super(BCG, self).__init__(new_size, crop_size, random_resize,
                                  is_flip, to_tensor, mean, std,)
        self.root = root
        self.mode = mode
        self.classes = classes
        # Read file paths:
        if root and mode:
            self.files = self.read_files()
        # Prepare dictionaries for convenient convertion:
        self.trainId2name = {label.trainId: label.name for label in labels}
        self.trainId2color = {label.trainId: label.color for label in labels}
        self.trainId2color[255] = (0, 0, 0)
        self.trainId2color.pop(-1)
        # Get new label mapping. This is different from the original trainId.
        self.label_mapping = self.get_label_mapping(classes)

        """self.class_weights = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345,
                                        1.0166, 0.9969, 0.9754, 1.0489,
                                        0.8786, 1.0023, 0.9539, 0.9843,
                                        1.1116, 0.9037, 1.0865, 1.0955,
                                        1.0865, 1.1529, 1.0507]).cuda()"""

    def read_files(self):
        """ Read and prepare file paths.

            Args:

            Return:
                files (list of dictionary of image file path & label file path)
        """
        files = []
        if 'test' in self.mode:
            imgs = sorted(glob.glob(os.path.join(self.root, 'images', self.mode, '*')))
            for img_path in imgs:
                files.append({
                    'images': img_path
                })
        else:
            imgs = sorted(glob.glob(os.path.join(self.root, 'images', self.mode, '*')))
            labs = sorted(glob.glob(os.path.join(self.root, 'labels', self.mode, '*')))
            for (img_path, lab_path) in zip(imgs, labs):
                files.append({
                    'image': img_path,
                    'label': lab_path
                })
        return files

    def get_label_mapping(self, classes):
        """ Get label mapping from original trainId to new Id.
            The new Id match with the specified classes.

            Args:
                classes (list of string): List of classes that are considered
                in the current model.

            Return:
                label_mapping (dictionary): Dictionary with key is original
                trainId, value is the new trainId.
        """
        label_mapping = dict()
        i = 0
        if classes is None:
            for trainId in set([label.trainId for label in labels]):
                label_mapping[trainId] = trainId
        else:
            for trainId in set([label.trainId for label in labels]):
                if self.trainId2name[trainId] in classes:
                    label_mapping[trainId] = i
                    i += 1
                else:
                    label_mapping[trainId] = len(classes)
        return label_mapping

    def convert_label(self, label, inverse=False):
        """ Original trainId <=> Custom trainId Convertion.

            Args:
                label (2d-array like): 2d array of trainId.
                inverse (bool): If True, convert from custom trainId to
                        original trainId. If False, convert original
                        trainId to custom trainId.

            Return:
                label (2d-array like): Converted trainId.
        """
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    def convert_color(self, input, inverse=False):
        """ Original trainId <=> Color Convertion.
            Labels have to be in original trainId.
            The colorscheme follows CityScapes convention.

            Args:
                input (2d-array like): 2d array of trainId / RGB color.
                inverse (bool): If True, convert from RGB color to original
                        trainId. If False, convert original trainId to 
                        custom trainId.

            Return:
                output (2d image): Final convertion.
        """
        if inverse:
            output = np.zeros(input.shape[:2], dtype=np.uint8)
            for v, k in self.trainId2color.items():
                output[(input == k).sum(-1) == 3] = v
            return output
        else:
            output = np.zeros((*input.shape, 3), dtype=np.uint8)
            for k, v in self.trainId2color.items():
                output[input == k] = v
            return output

    def __getitem__(self, index):
        """ Get item with index.

            Return:
                image, label (tuple of 2d-array like)
        """
        item = self.files[index]
        image = cv2.imread(item["image"], cv2.IMREAD_COLOR)
        size = image.shape
        label = cv2.imread(item["label"], cv2.IMREAD_GRAYSCALE)
        label = self.convert_label(label)

        image, label = self.gen_sample(image, label)

        return image, label
