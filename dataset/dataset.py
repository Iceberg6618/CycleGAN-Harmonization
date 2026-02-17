import os
import pickle
from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms

class Paired_Dataset(Dataset):
    def __init__(self,
                 data_root,
                 vendor1,
                 vendor2,
                 transforms=transforms.ToTensor(),
                 data_range=(0.1, 0.9)):
        
        self.vendor1, self.vendor2 = vendor1, vendor2
        
        self.dataset = []
        subj_list = os.listdir(data_root)
        for subj in subj_list:
            self.dataset.extend(self.generate_dataset(data_root, subj, data_range[0], data_range[1]))
            
        self.transforms = transforms
    
    def generate_dataset(self, root, subj, lower_bound, upper_bound):
        file_list = os.listdir(os.path.join(root, subj))
        data_length = len(file_list)
        lb, ub = int(data_length*lower_bound), int(data_length*upper_bound)
        return [os.path.join(root, subj, path) for path in file_list[lb:ub]]
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        with open(self.dataset[index], 'rb') as f:
            data = pickle.load(f)

        return {
            self.vendor1:self.transforms(data[self.vendor1]),
            self.vendor2:self.transforms(data[self.vendor2])
        }