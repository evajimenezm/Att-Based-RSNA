import torch

import numpy as np
import torchvision

from collections import deque

import pandas as pd


class MNIST_MILDataset(torch.utils.data.Dataset):
    def __init__(self, name, mode, num_bags, obj_labels, bag_size) -> None:
        super().__init__()
        self.name = name
        if self.name == 'mnist':
            self.dataset = torchvision.datasets.MNIST(root='./AttentionBased/data/', train=(mode=="train"), download=True)
            self.dataset.data = self.dataset.data.numpy().reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
        self.num_bags = num_bags
        self.obj_labels = obj_labels
        self.bag_size = bag_size

        self.bag_data = None
        self.bag_labels = None
        self.inst_labels = None

        self.create_bags()
    
    def create_bags(self):
        pos_idx = np.where(np.isin(self.dataset.targets, self.obj_labels))[0]
        np.random.shuffle(pos_idx)
        neg_idx = np.where(~np.isin(self.dataset.targets, self.obj_labels))[0]
        np.random.shuffle(neg_idx)

        p_positive = len(pos_idx) / (len(self.dataset.targets))

        num_pos_bags = self.num_bags // 2
        num_neg_bags = self.num_bags - num_pos_bags

        pos_idx_queue = deque(pos_idx)
        neg_idx_queue = deque(neg_idx)

        self.bag_data = []
        self.bag_labels = []
        self.inst_labels = []
        for i in range(num_pos_bags):
            bag = []
            y_labels = []
            #num_positives = np.random.randint(1, self.bag_size//2)
            num_positives = np.random.binomial(self.bag_size, p_positive)
            while num_positives == 0:
                num_positives = np.random.binomial(self.bag_size, p_positive)
            num_negatives = self.bag_size - num_positives
            for _ in range(num_positives):
                a = pos_idx_queue.pop()
                bag.append(self.dataset.data[a])
                y_labels.append(self.dataset.targets[a])
                pos_idx_queue.appendleft(a)
            for _ in range(num_negatives):
                a = neg_idx_queue.pop()
                bag.append(self.dataset.data[a])
                y_labels.append(self.dataset.targets[a])
                neg_idx_queue.appendleft(a)

            idx_sort = np.argsort(y_labels)
            bag = np.stack(bag)[idx_sort]
            y_labels = np.array(y_labels)[idx_sort]
            y_labels = np.where(np.isin(y_labels, self.obj_labels), 1, 0)
            bag_label = np.max(y_labels)

            self.bag_data.append(bag)
            self.bag_labels.append(bag_label)
            self.inst_labels.append(y_labels)

        for i in range(num_neg_bags):
            bag = []
            y_labels = []
            for _ in range(self.bag_size):
                a = neg_idx_queue.pop()
                bag.append(self.dataset.data[a])
                y_labels.append(self.dataset.targets[a])
                neg_idx_queue.appendleft(a)

            idx_sort = np.argsort(y_labels)
            bag = np.stack(bag)[idx_sort]
            y_labels = np.array(y_labels)[idx_sort]
            y_labels = np.zeros_like(y_labels)
            bag_label = 0

            self.bag_data.append(bag)
            self.bag_labels.append(bag_label)
            self.inst_labels.append(y_labels)
    
    def __len__(self):      
        return len(self.bag_data)
    
    def __getitem__(self, index):
        return torch.from_numpy(self.bag_data[index]), torch.tensor(self.bag_labels[index]), torch.from_numpy(self.inst_labels[index])





class RSNA_ICH_MILDataset(torch.utils.data.Dataset):
    def __init__(self, mode, validation = False, bag_size = 60) -> None:
        super().__init__()
        self.bag_size = bag_size
        self.mode = mode
        self.validation = validation

        self.bag_data = None
        self.bag_labels = None
        self.inst_labels = None

        self.load_bags() 
    
    def load_bags(self):

        # Cargamos el archivo CSV
        bags_df = pd.read_csv(f'/data/datasets/RSNA_ICH/bags_{self.mode}.csv')  
        dataset_name = 'Test'

        # Definimos las variables a rellenar
        bag_names = []
        self.bag_data = []
        self.bag_labels = []
        self.inst_labels = []

        # Comenzamos a rellenar las variables
        bag_names = np.array(bags_df['bag_name'].drop_duplicates())
        bag_names = bag_names[:int(len(bag_names)*0.03)]
        pos_ini = 0
        pos_fin = len(bag_names)

        if self.mode == 'train':
            if self.validation:
                pos_ini = int(0.8*len(bag_names))
                dataset_name = 'Validation'
            else:
                pos_fin = int(0.8*len(bag_names))
                dataset_name = 'Train'

        print(f'Cargando el dataset "{dataset_name}"...')

        for i in range(pos_ini, pos_fin):
            print(f'Leyendo bolsa número {i+1}, con nombre {bag_names[i]}...')

            # Tomo el subset correspondiente a la bolsa en la que nos encontramos
            bag_df = bags_df.loc[bags_df['bag_name'] == bag_names[i]]

            # Extraigo las propiedades que nos van a hacer falta
            bag_instance_labels = np.array(bag_df['instance_label'])
            bag_instance_names = np.array(bag_df['instance_name'])
            print(f'El nombre de los cortes son {bag_instance_names}')
            bag_label = bag_df.iloc[0]['bag_label']

            # Cargo las imágenes de la bolsa
            images_data = []
            for j in range(len(bag_instance_labels)):
                try:
                    image = np.load(f'/data/datasets/RSNA_ICH/{self.mode}/{bag_instance_names[j].split(".")[0]}.npy', allow_pickle=True)
                    images_data.append(image)
                except:
                    print(f'La slice {bag_instance_names[j]} no se ha encontrado en el la carpeta, así que eliminamos la etiqueta asociada a la misma.')
                    bag_instance_labels = np.delete(bag_instance_labels, j)

            if len(images_data) < self.bag_size:
                num_slices_original = len(images_data)
                imagen_nueva = np.full(images_data[0].shape, 0.)
                images_data.extend([imagen_nueva] * (self.bag_size-num_slices_original))
                bag_instance_labels = np.append(bag_instance_labels, [0] * (self.bag_size-num_slices_original))

            self.bag_data.append(images_data)
            self.bag_labels.append(bag_label)
            self.inst_labels.append(bag_instance_labels)

            print(f'La bolsa {bag_names[i]}, con etiqueta {bag_label} ha terminado de leerse, y en ella hay {len(images_data)} slices y {len(bag_instance_labels)} etiquetas.')
        
        self.bag_data = np.array(self.bag_data, dtype=np.float32).transpose(0,1,4,2,3)
        print(torch.from_numpy(self.bag_data).size())
        self.bag_labels = np.array(self.bag_labels, dtype=np.float32)
        self.inst_labels = np.array(self.inst_labels, dtype=np.float32)
            

    
    def __len__(self):      
        return len(self.bag_data)
    
    def __getitem__(self, index):
        return torch.from_numpy(self.bag_data[index]), torch.tensor(self.bag_labels[index]), torch.from_numpy(self.inst_labels[index])