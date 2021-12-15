'''
Created on Sept, 2019, based on PyTorch open source implementation
@author: Julio A Martinez
'''

from utils.vision import VisionDataset

from PIL import Image

import os
import os.path
import sys
import numpy as np
import random


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)

def make_dataset(class_to_idx, max_samples, maxout, read_seed=None, extensions=None, is_valid_file=None):
    '''
    Input:
        max_samples: the max number of samples per category (Note: allows for less and therfore a non
                                                            uniform number of samples per category. 
                                                            Can easily be changed to force uniformity):
    Return:
        images: list of image paths sorted lexigraphically by name
    '''
    print('\nread_seed:', read_seed)
    
    images = []
    classes_temp = []
    images_dict = {}
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    
    #print(class_to_idx)
    for target in sorted(class_to_idx.keys()):
        images_dict[target] = []
        
        #d = os.path.join(dir, target)
        if not os.path.isdir(target):
            continue

        num_samples_reached = 0
        for root, _, fnames in sorted(os.walk(target)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                task_name = os.path.join(os.path.join(os.path.join(path, os.pardir),os.pardir), os.pardir)
                #print('task_name 1:', task_name)
                task_name = os.path.abspath(task_name)
                #print('task_name 2:', task_name)
                task_name = os.path.basename(task_name)
                #print('task_name 3:', task_name)
                if is_valid_file(path):
                    if (max_samples[task_name] is None):
                        item = (path, class_to_idx[target])
                        images.append(item)
                    elif maxout: # continues until last image is taken
                        #print('maxout')
                        item = (path, class_to_idx[target], task_name)
                        #print(type(class_to_idx[target]))
                        #images.append(item)
                        
                        #print('target:', target)
                        #print('item:', item)
                        
                        item = (path, class_to_idx[target])
                        images_dict[target].append(item)
                        if (target,'task_name') not in images_dict:
                            images_dict[(target,'task_name')] = task_name
                       
                    elif num_samples_reached < max_samples[task_name]:
                        #print('no maxout')
                        item = (path, class_to_idx[target])
                        images.append(item)
                        num_samples_reached +=1
                    else:
                        break
            if maxout:
                temp  = np.array(images_dict[target])  # column one is abs path, column two is idx, column three is task
                image_paths_temp, image_idx_temp = temp[:,0], temp[:,1]

                permutation = np.arange(len(image_paths_temp))
                
                if read_seed is not None:
                    np.random.seed(read_seed)
                np.random.shuffle(permutation) # in place

                image_paths_temp, image_idx_temp = image_paths_temp[permutation], image_idx_temp[permutation]
                argsort = np.argsort(image_paths_temp[:max_samples[task_name]])
                image_paths_temp = image_paths_temp[argsort].tolist() 
                image_idx_temp = image_idx_temp[argsort].astype(int).tolist()
                
                temp = tuple(zip(image_paths_temp, image_idx_temp))
                #print(temp)

                images.extend(temp)
                classes_temp.extend(image_idx_temp)
    
    if maxout:
        unique_elements, counts_elements = np.unique(classes_temp, return_counts=True)
        print("\nFrequency of classes:")
        print(np.asarray((unique_elements, counts_elements)))
        print()
        
        
    return images

class DatasetFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::
        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext
        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid_file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, loader, extensions=None, transform=None,
                 target_transform=None, is_valid_file=None, max_samples=None, maxout=True, read_seed=None, includePaths=False):
        super(DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(class_to_idx, max_samples, maxout, read_seed, extensions, is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root[0] + ' and ' + self.root[1] + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.includePaths = includePaths
        self.loader = loader
        self.extensions = extensions
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_classes(self, dirs):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        dirs.sort()
        #print(dirs)
        task_to_num_classes = {}
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            #classes = [d.name for d in os.scandir(dir) if d.is_dir()]
            #classes = [os.path.abspath(d) for d in os.scandir(dir) if d.is_dir()]
            classes = []
            for dir in dirs:
                classes_temp = [os.path.abspath(d) for d in os.scandir(dir) if d.is_dir()]
                task_name = os.path.basename(os.path.abspath(os.path.join(dir, os.pardir)))
                task_to_num_classes[task_name] = len(classes_temp)
                #classes.extend([os.path.abspath(d) for d in os.scandir(dir) if d.is_dir()])
                classes.extend(classes_temp)  
        else:
            #classes = [os.path.abspath(d) for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
            classes = []
            for dir in dirs:
                classes_temp = [os.path.abspath(d) for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
                #classes.extend([os.path.abspath(d) for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))])
                task_name = os.path.basename(os.path.abspath(os.path.join(dir, os.pardir)))
                task_to_num_classes[task_name] = len(classes_temp)
                classes.extend(classes_temp)
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.task_to_num_classes = task_to_num_classes
        
        #print('classes:',classes)
        #print('class_to_idx:', class_to_idx)
        
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        if self.includePaths:
            return (sample, target, path)
        else:
            return sample, target

    def __len__(self):
        return len(self.samples)


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


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


class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid_file (used to check of corrupt files)
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None, max_samples=None, read_seed=None, maxout=True, includePaths=False):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file,
                                          max_samples=max_samples,
                                          maxout=maxout,
                                          read_seed=read_seed,
                                          includePaths=includePaths)
        self.imgs = self.samples
