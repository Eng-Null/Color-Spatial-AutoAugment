import torchvision
import torchvision.transforms as transforms
import numpy as np

from torch.utils.data import DataLoader, Subset
from augmention.auto_augment import AutoAugment, Cutout
from augmention.cs_auto_augment import CS_AutoAugment
from torchvision.datasets import CIFAR100, CIFAR10
from PIL import Image

def get_mean_std(args):   
    if args.dataset == 'cifar10' or args.dataset == 'cifar10_reduced':
        data = torchvision.datasets.CIFAR10(root='data', train=True, download=True)
        x = np.concatenate([np.asarray(data[i][0]) for i in range(len(data))])
        mean = np.mean(x, axis=(0,1)) / 255
        std = np.std(x, axis=(0,1)) / 255
        return mean, std
    
    elif args.dataset == 'cifar100' or args.dataset == 'cifar100_reduced':
        data = torchvision.datasets.CIFAR100(root='data', train=True, download=True)
        x = np.concatenate([np.asarray(data[i][0]) for i in range(len(data))])
        mean = np.mean(x, axis=(0,1)) / 255
        std = np.std(x, axis=(0,1)) / 255
        return mean, std
    else:  
        raise ValueError('not supported dataset "%s"' % (args.dataset))

class CIFAR10Pair(CIFAR10):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        # Apply the same transformation to both versions of the image
        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target

class CIFAR100Pair(CIFAR100):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        # Apply the same transformation to both versions of the image
        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target

    
class Transform_train_Cifar10:
    def __init__(self, args):
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

        transform_train = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]

        if args.augment == 'AutoAugment':
            transform_train.append(AutoAugment())
        elif args.augment == 'Augment':
            transform_train.append(CS_AutoAugment())
        elif args.augment == 'Basic':
            transform_train.extend([
                transforms.RandomApply([transforms.ColorJitter(0.3, 0.3, 0.3, 0.1)], 0.8),
                transforms.RandomRotation(2),
                transforms.RandomAffine(degrees=0, shear=8),
                transforms.RandomGrayscale(0.1),
                ])
        #if args.cutout:
            #transform_train.append(Cutout())
        mean, std = get_mean_std(args)
        
        transform_train.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean,
                                    std),
        ])

        self.transform_train = transforms.Compose(transform_train)

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean,
                                    std),
        ])

    def load_data(self):
        if (self.dataset == 'cifar10' or self.dataset == 'cifar10_reduced'):
            train_data = CIFAR10Pair(root='data', train=True, transform=self.transform_train, download=True)
            test_data = CIFAR10Pair(root='data', train=False, transform=self.transform_test, download=True)

            if self.dataset == 'cifar10':
                train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, drop_last=True)         
            else:
                # Define the number of images to select from each class
                num_images_per_class = 400

                # Select a subset of the images from each class
                selected_indices = []
                for i in range(10):
                    indices = np.where(np.array(train_data.targets) == i)[0][:num_images_per_class]
                    selected_indices.extend(indices)
                
                train_subset = Subset(train_data, selected_indices)
                train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, drop_last=True)
            
            test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
            
            return train_loader, test_loader
        
        elif(self.dataset == 'cifar100' or self.dataset == 'cifar100_reduced'):
            train_data = CIFAR100Pair(root='data', train=True, transform=self.transform_train, download=True)
            test_data = CIFAR100Pair(root='data', train=False, transform=self.transform_test, download=True)
            
            if self.dataset == 'cifar100':
                train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, drop_last=True)
            else:
                # Define the number of images to select from each class
                num_images_per_class = 400

                # Select a subset of the images from each class
                selected_indices = []
                for i in range(10):
                    indices = np.where(np.array(train_data.targets) == i)[0][:num_images_per_class]
                    selected_indices.extend(indices)
                
                train_subset = Subset(train_data, selected_indices)
                train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, drop_last=True)

            test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
            return train_loader, test_loader

    
    def load_Single_data(self):
        if (self.dataset == 'cifar10' or self.dataset == 'cifar10_reduced'):
            train_data = CIFAR10(root='data', train=True, transform=self.transform_train, download=True)
            test_data = CIFAR10(root='data', train=False, transform=self.transform_test, download=True)

            if self.dataset == 'cifar10':
                train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, drop_last=True)         
            else:
                # Define the number of images to select from each class
                num_images_per_class = 400

                # Select a subset of the images from each class
                selected_indices = []
                for i in range(10):
                    indices = np.where(np.array(train_data.targets) == i)[0][:num_images_per_class]
                    selected_indices.extend(indices)
                
                train_subset = Subset(train_data, selected_indices)
                train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, drop_last=True)
            
            test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
            
            return train_loader, test_loader
        
        elif(self.dataset == 'cifar100' or self.dataset == 'cifar100_reduced'):
            train_data = CIFAR100(root='data', train=True, transform=self.transform_train, download=True)
            test_data = CIFAR100(root='data', train=False, transform=self.transform_test, download=True)
            
            if self.dataset == 'cifar100':
                train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, drop_last=True)
            else:
                # Define the number of images to select from each class
                num_images_per_class = 400

                # Select a subset of the images from each class
                selected_indices = []
                for i in range(10):
                    indices = np.where(np.array(train_data.targets) == i)[0][:num_images_per_class]
                    selected_indices.extend(indices)
                
                train_subset = Subset(train_data, selected_indices)
                train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, drop_last=True)

            test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
            return train_loader, test_loader

        