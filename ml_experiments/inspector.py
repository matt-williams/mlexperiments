import math
import torch

class Inspector:
    def __init__(self, model, inputs_to_track=[], outputs_to_track=[]):
        self.model = model
        self.input_hooks = {}
        self.output_hooks = {}
        self.inputs = {}
        self.outputs = {}
        self.track_inputs(*inputs_to_track)
        self.track_outputs(*outputs_to_track)

    def track_inputs(self, *inputs):
        self.untrack_outputs(inputs)
        for input in inputs:
            modules = [module for name, module in self.model.named_modules() if name == input]
            if len(modules) == 0:
                raise ValueError("No module {} to track input".format(input))
            module = modules[0]
            hook = module.register_forward_pre_hook(self.__input_hook_fn(input))
            self.input_hooks[input] = hook
            self.inputs[input] = None

    def track_outputs(self, *outputs):
        self.untrack_outputs(outputs)
        for output in outputs:
            modules = [module for name, module in self.model.named_modules() if name == output]
            if len(modules) == 0:
                raise ValueError("No module {} to track output".format(output))
            module = modules[0]
            hook = module.register_forward_hook(self.__output_hook_fn(output))
            self.output_hooks[output] = hook
            self.outputs[output] = None

    def untrack_inputs(self, *inputs):
        for input in inputs:
            if input in self.input_hooks:
                self.input_hooks[input].remove()
                del self.input_hooks[input]
                del self.inputs[input]
                
    def untrack_outputs(self, *outputs):
        for output in outputs:
            if output in self.output_hooks:
                self.output_hooks[output].remove()
                del self.output_hooks[output]
                del self.outputs[output]

    def untrack_all(self):
        self.untrack_inputs(self.input_hooks.keys())
        self.untrack_outputs(self.output_hooks.keys())

    def reset(self):
        for input in self.inputs:
            self.inputs[input] = None
        for output in self.outputs:
            self.outputs[output] = None
    
    def __input_hook_fn(self, name):
        def hook_fn(module, x):
            if self.inputs[name] is None:
                self.inputs[name] = x[0].data.clone()
            else:
                self.inputs[name] = torch.cat([self.inputs[name], x[0].data], dim=0)
        return hook_fn

    def __output_hook_fn(self, name):
        def hook_fn(module, x, y):
            if self.outputs[name] is None:
                self.outputs[name] = y.data.clone()
            else:
                self.outputs[name] = torch.cat([self.outputs[name], y.data], dim=0)
        return hook_fn


def svd_entropy(x):
    x = x - x.mean(0)
    x = x / x.var(dim=0).sqrt()
    x = x.view(x.shape[0], -1)
    _, S, _ = torch.svd(torch.t(x))
    S = S / S.sum()
    entropy = -((S * S.log()).sum()) / math.log(2)
    return entropy

def svd_entropy_per_sample(x):
    num_bits = x.shape[0] / math.log(2)
    return svd_entropy(x) / num_bits

# Not sure if this is useful
def svd_entropy_per_bit(x):
    shape = x.view(x.shape[0], -1).shape
    num_bits = math.log(min(shape[0], shape[1])) / math.log(2)
    return svd_entropy(x) / num_bits