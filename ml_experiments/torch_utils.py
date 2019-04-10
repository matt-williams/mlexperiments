import math
import copy
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn


class SnapshottableOptimizer:
    def __init__(self, optimizer, snapshot_interval=None, max_snapshots=3, save_snapshots=False):
        self.optimizer = optimizer
        self.snapshot_interval = snapshot_interval
        self.max_snapshots = max_snapshots
        self.save_snapshots = save_snapshots
        self.snapshot_countdown = snapshot_interval
        self.snapshots = []

    def step(self, *args, **kwargs):
        result = self.optimizer.step(*args, **kwargs)
        if self.snapshot_countdown:
            self.snapshot_countdown -= 1
            if self.snapshot_countdown <= 0:
                self.snapshot()
        return result

    def snapshot(self):
        params = [param for param in self.optimizer.__getstate__()["state"].keys()]
        self.snapshots = self.snapshots[1-self.max_snapshots:] if self.max_snapshots > 1 else []
        self.snapshots.append(copy.deepcopy(params))
        self.snapshot_countdown = self.snapshot_interval

    def rewind(self, index=0):
        index = len(self.snapshots) - 1 - index if index >= 0 and index < len(self.snapshots) else -index
        snapshot = self.snapshots[index]
        params = [param for param in self.optimizer.__getstate__()["state"].keys()]
        for param, snapshot_param in zip(params, snapshot):
            param.data.set_(snapshot_param.data.clone())

    def get_num_snapshots(self):
        return len(self.snapshots)

    def __getstate__(self):
        state = self.optimizer.__getstate__()
        if self.save_snapshots:
            state["snapshots"] = copy.deepcopy(self.snapshots)
        return state

    def __setstate__(self, state):
        self.snapshots = []
        return self.optimizer.__setstate__(state)

    def add_param_group(self, param_group):
        return self.optimizer.add_param_group(param_group)

    def load_state_dict(self, state_dict):
        if "snapshots" in state_dict:
            self.snapshots = state_dict["snapshots"]
        return self.optimizer.load_state_dict(state_dict)

    def state_dict(self):
        state = self.optimizer.state_dict()
        if self.save_snapshots:
            state["snapshots"] = copy.deepcopy(self.snapshots)
        return state

    def zero_grad(self):
        return self.optimizer.zero_grad()


class ObservableModule(nn.Module):
    def __init__(self, model, outputs_to_observe=[], inputs_to_observe=[], reset_on_mode_change=True):
        super(ObservableModule, self).__init__()
        self.model = model
        self.__train_mode = True
        self.__reset_on_mode_change = reset_on_mode_change
        self.__inputs = {}
        self.__outputs = {}
        self.observe_outputs(*outputs_to_observe)
        self.observe_inputs(*inputs_to_observe)

    def __get_module(self, name):
        modules = [module for name2, module in self.model.named_modules() if name2 == name]
        if len(modules) == 0:
            raise ValueError("No module {} to observe".format(name))
        return modules[0]

    def observe_inputs(self, *inputs):
        for input in inputs:
            self.__get_module(input) # to check it exists
            self.__inputs[input] = None

    def observe_outputs(self, *outputs):
        for output in outputs:
            self.__get_module(output) # to check it exists
            self.__outputs[output] = None

    def unobserve_inputs(self, *inputs):
        for input in inputs:
            if input in self.__inputs:
                del self.__inputs[input]

    def unobserve_outputs(self, *outputs):
        for output in outputs:
            if output in self.__outputs:
                del self.__outputs[output]

    def unobserve_all(self):
        self.__inputs = {}
        self.__outpus = {}

    def reset(self):
        for input in self.__inputs:
            self.__inputs[input] = None
        for output in self.__outputs:
            self.__outputs[output] = None

    def get_observed_input(self, input):
        return self.__inputs[input]

    def get_observed_output(self, output):
        return self.__outputs[output]

    def __input_hook_fn(self, name):
        def hook_fn(module, x):
            if self.__inputs[name] is None:
                self.__inputs[name] = x[0].data.clone()
            else:
                self.__inputs[name] = torch.cat([self.__inputs[name], x[0].data], dim=0)
        return hook_fn

    def __output_hook_fn(self, name):
        def hook_fn(module, x, y):
            if self.__outputs[name] is None:
                self.__outputs[name] = y.data.clone()
            else:
                self.__outputs[name] = torch.cat([self.__outputs[name], y.data], dim=0)
        return hook_fn

    def train(self, mode=True):
        if not self.__train_mode == mode:
            self.reset()
        self.__train_mode = mode
        return super(ObservableModule, self).train(mode)

    def eval(self):
        if self.__train_mode:
            self.reset()
        self.__train_mode = False
        return super(ObservableModule, self).eval()

    def forward(self, *input):
        hooks = []
        for input2 in self.__inputs:
            module = self.__get_module(input2)
            hook = module.register_forward_pre_hook(self.__input_hook_fn(input2))
            hooks.append(hook)
        for output2 in self.__outputs:
            module = self.__get_module(output2)
            hook = module.register_forward_hook(self.__output_hook_fn(output2))
            hooks.append(hook)
        try:
            output = self.model.forward(*input)
        except:
            for hook in hooks:
                hook.remove()
            raise
        for hook in hooks:
            hook.remove()
        return output


# TODO: Find a better place for these - they're best used with ObservableModule, but not really relevant otherwise
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


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Residual(nn.Sequential):
    def __init__(self, *args):
        super(Residual, self).__init__(*args)

    def forward(self, x):
        x = x + super(Residual, self).forward(x)
        return x

class Downsample(nn.MaxPool2d):
    def __init__(self, resolution, scale):
        super(Downsample, self).__init__(2, padding=[math.ceil(x / scale) % 2 for x in resolution])

class Crop2d(nn.Module):
    def __init__(self, shape):
        super(Crop2d, self).__init__()
        self.shape = shape

    def forward(self, x):
        x = x[:, :, :self.shape[0], :self.shape[1]]
        return x

class Parallel(nn.Module):
    def __init__(self, *inner):
        super(Parallel, self).__init__()
        self.inner = inner
        for ii, inner in enumerate(self.inner):
            self.add_module(str(ii), inner)

    def forward(self, x):
        x = torch.cat([inner(x) for inner in self.inner], dim=1)
        return x

class Upsample(nn.Sequential):
    def __init__(self, resolution, scale):
        super(Upsample, self).__init__(
            nn.Upsample(scale_factor=2, mode='bilinear'), # , align_corners=True), # not supported in my version of PyTorch
            Crop2d([math.ceil(x / scale) for x in resolution]),
        )

class ConvDownsample(nn.Sequential):
    def __init__(self, ch, resolution, scale):
        super(Downsample, self).__init__(
            nn.Conv2d(ch, ch, 4, padding=tuple([1 + math.ceil(x / scale) % 2 for x in resolution]), stride=2),
            nn.LeakyReLU(inplace=True),
        )

class ConvUpsample(nn.Sequential):
    def __init__(self, ch, resolution, scale):
        super(Upsample, self).__init__(
            nn.ConvTranspose2d(ch, ch, 4, padding=1, stride=2),
            Crop2d([math.ceil(x / scale) for x in resolution]),
            nn.LeakyReLU(inplace=True),
        )

class View(nn.Module):
    def __init__(self, *view):
        super(View, self).__init__()
        self.view = view

    def forward(self, x):
        x = x.view(*self.view)
        return x

class RandomFeaturesAppender(nn.Module):
    def __init__(self, features=1, power=1):
        super(RandomFeaturesAppender, self).__init__()
        self.features = features
        self.power = power

    def forward(self, x, sigma=1):
        if self.features > 0:
            sigma = sigma * self.power / math.sqrt((np.array(x.shape[1:]) * np.array(x.shape[1:])).sum())
            rand_features = torch.randn(*((x.shape[0], self.features) + x.shape[2:]))
#            rand_features = rand_features.cuda() if CUDA else rand_features
            x = torch.cat([x, Variable(rand_features * sigma, requires_grad=False)], dim=1)
        return x

class AddConstant(nn.Module):
    def __init__(self, constant):
        super(AddConstant, self).__init__()
        self.constant = constant

    def forward(self, x):
        x = x + self.constant
        return x

class ExpClamp(nn.Module):
    def __init__(self, min, max, alpha=1.0):
        super(ExpClamp, self).__init__()
        self.min = min
        self.max = max
        self.alpha = alpha

    def forward(self, x):
        x = torch.clamp(x, self.min, self.max) + \
            self.alpha * (torch.clamp(torch.exp(x - self.min) - 1.0, -1.0, 0.0) + \
                          torch.clamp(1.0 - torch.exp(self.max - x), 0.0, 1.0))
        return x

class LeakyClamp(nn.Module):
    def __init__(self, min, max, alpha=0.01):
        super(LeakyClamp, self).__init__()
        self.min = min
        self.max = max
        self.alpha = alpha

    def forward(self, x):
        FLOAT_TYPE = type(x.data)
        x = torch.clamp(x, self.min, self.max) + \
            self.alpha * (torch.min(x - self.min, Variable(FLOAT_TYPE([0.0]))) + \
                          torch.max(x - self.max, Variable(FLOAT_TYPE([0.0]))))
        return x

class ExpMeanClamp(nn.Module):
    def __init__(self, mean_delta=-1, alpha=1.0):
        super(ExpMeanClamp, self).__init__()
        self.mean_delta = mean_delta
        self.alpha = alpha

    def forward(self, x):
        mean_delta = (x.view(x.shape[0], -1).mean(1, keepdim=True) + self.mean_delta).view(-1, *[1 for _ in x.shape[1:]])
        x = x - mean_delta
        x = nn.functional.elu(x, self.alpha, inplace=True)
        x = x + mean_delta
        return x

class AddCoords(nn.Module):
    def forward(self, x):
        xs = x.shape
        x1 = torch.arange(-1.0, 1.0 + 1.0 / (xs[2] - 1), 2.0 / (xs[2] - 1)).view((1, 1, xs[2], 1)).expand((xs[0], 1) + xs[2:])
        x2 = torch.arange(-1.0, 1.0 + 1.0 / (xs[3] - 1), 2.0 / (xs[3] - 1)).view((1, 1, 1, xs[3])).expand((xs[0], 1) + xs[2:])
        x = torch.cat([x, x1, x2], dim=1)
        return x

class Fire(nn.Sequential):
    def __init__(self, in_ch, out_ch, s1x1=None, e1x1=None, s1x1_in_ch_ratio=1.0, e1x1_out_ch_ratio=0.5, do_last_relu=True):
        s1x1 = s1x1 or int(round(in_ch * s1x1_in_ch_ratio))
        e1x1 = e1x1 or int(round(out_ch * e1x1_out_ch_ratio))
        e3x3 = out_ch - e1x1
        super(Fire, self).__init__(
            nn.Conv2d(in_ch, s1x1, 1),
            Parallel(
                nn.Conv2d(s1x1, e1x1, 1),
                nn.Conv2d(s1x1, e3x3, 3, padding=1),
            ),
            nn.LeakyReLU(inplace=True) if do_last_relu else Identity()
        )

class ResidualFireStack(nn.Sequential):
    def __init__(self, ch, in_ch=None, out_ch=None, norm=nn.BatchNorm2d):
        in_ch = in_ch or ch
        out_ch = out_ch or ch
        super(ResidualFireStack, self).__init__(
            nn.Conv2d(in_ch, ch, 1) if in_ch != ch else Identity(),
            Residual(nn.Sequential(
                Fire(ch, ch),
                norm(ch),
                Fire(ch, ch),
                norm(ch),
                Fire(ch, ch, do_last_relu=False),
                #norm(ch), # TODO: Decide whether this is appropriate.
            )),
            nn.Conv2d(ch, out_ch, 1) if out_ch != ch else Identity(),
            nn.LeakyReLU(inplace=True),
            norm(out_ch),
        )

class FireDown(nn.Sequential):
    def __init__(self, in_ch, resolution, scale, out_ch, norm=nn.BatchNorm2d):
        super(FireDown, self).__init__(
            ConvDownsample(in_ch, resolution, scale), # TODO: Should this go after the fire stack?
            ResidualFireStack(out_ch, in_ch=in_ch, norm=norm),
        )

class FireUp(nn.Sequential):
    def __init__(self, in_ch, resolution, scale, out_ch, norm=nn.BatchNorm2d):
        super(FireUp, self).__init__(
            ConvUpsample(in_ch, resolution, scale),
            ResidualFireStack(in_ch, out_ch=out_ch, norm=norm),
        )
