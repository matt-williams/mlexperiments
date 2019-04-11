import os
import pickle
import torch
import logging as lg

class Session:
    def __init__(self, file_prefix, torchable_types=(torch.nn.modules.module.Module, torch.optim.Optimizer), logger_types=(lg.Logger), cuda=False):
        super(Session, self).__init__()
        self.file_prefix = file_prefix
        self.torchable_types = torchable_types
        self.logger_types = logger_types
        self.cuda = cuda
        self.state = {}

    def is_torchable(self, value):
        return isinstance(value, self.torchable_types)
        
    def is_logger(self, value):
        return isinstance(value, self.logger_types)
        
    def is_picklable(self, value):
        return isinstance(value, list)
        
    def __getitem__(self, key):
        return self.state[key]

    def __setitem__(self, key, value):
        if not self.is_torchable(value) and not self.is_logger(value) and not self.is_picklable(value):
            raise ValueError("Invalid session value for key {}".format(key))
        self.state[key] = value

    def __delitem__(self, key):
        del self.state[key]

    def get_filenames(self, file_prefix2=None):
        file_prefix = self.file_prefix + ('-' + file_prefix2 if file_prefix2 else '')
        filenames = []
        for key in self.state:
            value = self.state[key]
            if self.is_torchable(value):
                filenames.append('{}-{}.pt'.format(file_prefix, key))
            elif self.is_logger(value):
                filenames.append('{}-{}.pkl-log'.format(file_prefix, key))
            elif self.is_picklable(value):
                filenames.append('{}-{}.pkl'.format(file_prefix, key))
            else:
                print("Ignoring invalid session state: {}".format(key))
        return filenames
    
    def load(self, file_prefix2=None):
        file_prefix = self.file_prefix + ('-' + file_prefix2 if file_prefix2 else '')
        for key in self.state:
            value = self.state[key]
            if self.is_torchable(value):
                fname = '{}-{}.pt'.format(file_prefix, key)
                if os.path.exists(fname):
                    state_dict = torch.load(fname, map_location=lambda s, _: s.cuda(0) if self.cuda else s.cpu())
                    value.load_state_dict(state_dict)
            elif self.is_logger(value):
                fname = '{}-{}.pkl-log'.format(file_prefix, key)
                if os.path.exists(fname):
                    value.load(fname)
            elif self.is_picklable(value):
                fname = '{}-{}.pkl'.format(file_prefix, key)
                if os.path.exists(fname):
                    with open(fname, 'rb') as file:
                        self.state[key] = pickle.Unpickler(file).load()
            else:
                print("Ignoring invalid session state: {}".format(key))

    def save(self, file_prefix2=None):
        file_prefix = self.file_prefix + ('-' + file_prefix2 if file_prefix2 else '')
        for key in self.state:
            value = self.state[key]
            if self.is_torchable(value):
                torch.save(value.state_dict(), '{}-{}.pt'.format(file_prefix, key))
            elif self.is_logger(value):
                value.save('{}-{}.pkl-log'.format(file_prefix, key))
            elif self.is_picklable(value):
                with open('{}-{}.pkl'.format(file_prefix, key), 'wb') as file:
                    pickle.Pickler(file).dump(value)
            else:
                print("Ignoring invalid session state: {}".format(key))
