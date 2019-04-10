from torch.utils.data import Dataset

class TransformedDataset(Dataset):
    def __init__(self, dataset, transforms=[]):
        super(TransformedDataset, self).__init__()
        self.dataset = dataset
        self.transforms = [transforms] if not isinstance(transforms, (list, tuple)) else transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        for transform in self.transforms:
            transform(sample)
        return sample

class PartitionedDataset(Dataset):
    def __init__(self, dataset, partition=(0., 1.), seed=1):
        super(PartitionedDataset, self).__init__()
        self.dataset = dataset
        all_indices = [idx for idx in range(0, len(dataset))]
        rand = random.Random()
        rand.seed(seed)
        rand.shuffle(all_indices)
        self.indices = all_indices[math.floor(partition[0] * len(all_indices)):math.floor(partition[1] * len(all_indices))]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if idx >= len(self.indices):
            raise IndexError
        sample = self.dataset[self.indices[idx]]
        return sample

class ClassPartitionedDataset(Dataset):
    def __init__(self, dataset, class_col, partition=(0., 1.), seed=1):
        super(ClassPartitionedDataset, self).__init__()
        self.dataset = dataset
        self.indices = []
        rand = random.Random()
        rand.seed(seed)
        for col_value, rows in pd.DataFrame.from_records(list(dataset)).groupby(class_col)[class_col]:
            row_indices = [idx for idx in rows.index]
            rand.shuffle(row_indices)
            self.indices += row_indices[math.floor(partition[0] * len(row_indices)):math.floor(partition[1] * len(row_indices))]
        rand.shuffle(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if idx >= len(self.indices):
            raise IndexError
        return self.dataset[self.indices[idx]]
    
class CachedDataset(Dataset):
    def __init__(self, dataset, preload=False):
        super(CachedDataset, self).__init__()
        self.dataset = dataset
        self.cache = []
        if preload:
            for idx in len(dataset):
                self.cache[idx] = dataset[idx]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if not idx in self.cache:
            self.cache[idx] = self.dataset[idx]
        return self.cache[idx].copy()

class PandasDataset(Dataset):
    def __init__(self, data_frame):
        super(PandasDataset, self).__init__()
        self.data_frame = data_frame

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if idx >= len(self.data_frame):
            raise IndexError
        sample = {}
        for col in self.data_frame.columns:
            sample[col] = self.data_frame.at[idx, col]
        return sample

class DirectoryDataset(Dataset):
    def __init__(self, path):
        super(DirectoryDataset, self).__init__()
        self.path = path
        self.files = os.listdir(path)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return {"fname": self.files[idx]}