"""Hardware-Aware Analog Layers (Linear, Conv2d, LSTM)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
class AnalogLayerMixin:
    def apply_analog_physics(self, weight, training):
        if training:
            w_eff = weight
            if self.physics: w_eff = self.physics.apply_programming_noise(w_eff)
            return w_eff * self.alpha
        else:
            w_drifted = weight
            correction = 1.0
            if self.physics and self.t_inference > 1.0:
                w_drifted = self.physics.apply_drift(w_drifted, self.t_inference)
                if self.use_gdc: correction = self.physics.get_gdc_factor(self.t_inference)
            return (w_drifted * self.alpha) * correction
class AnalogLinear(nn.Linear, AnalogLayerMixin):
    def __init__(self, in_features, out_features, bias=True, physics_engine=None):
        super().__init__(in_features, out_features, bias=bias)
        self.physics = physics_engine
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.t_inference = 0.0; self.use_gdc = False
    def forward(self, input):
        return F.linear(input, self.apply_analog_physics(self.weight, self.training), self.bias)
class AnalogConv2d(nn.Conv2d, AnalogLayerMixin):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, physics_engine=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias)
        self.physics = physics_engine
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.t_inference = 0.0; self.use_gdc = False
    def forward(self, input):
        return F.conv2d(input, self.apply_analog_physics(self.weight, self.training), self.bias, self.stride, self.padding)
class AnalogLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, physics_engine=None, dropout=0.0):
        super().__init__()
        self.num_layers = num_layers; self.batch_first = batch_first; self.dropout = nn.Dropout(dropout)
        self.cells = nn.ModuleList([AnalogLSTMCell(input_size if i==0 else hidden_size, hidden_size, bias, physics_engine) for i in range(num_layers)])
    def forward(self, x, state=None):
        if self.batch_first: x = x.transpose(0, 1)
        if state is None: state = (torch.zeros(self.num_layers, x.size(1), self.cells[0].hidden_size, device=x.device), torch.zeros(self.num_layers, x.size(1), self.cells[0].hidden_size, device=x.device))
        h_prev, c_prev = state; outputs = []
        for t in range(x.size(0)):
            x_t = x[t]; h_t_layers, c_t_layers = [], []
            for i, cell in enumerate(self.cells):
                h_i, c_i = cell(x_t, (h_prev[i], c_prev[i]))
                h_t_layers.append(h_i); c_t_layers.append(c_i)
                x_t = self.dropout(h_i) if i < self.num_layers - 1 else h_i
            h_prev, c_prev = torch.stack(h_t_layers), torch.stack(c_t_layers); outputs.append(x_t)
        out = torch.stack(outputs)
        return (out.transpose(0, 1), (h_prev, c_prev)) if self.batch_first else (out, (h_prev, c_prev))
class AnalogLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, physics_engine=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.ih = AnalogLinear(input_size, 4 * hidden_size, bias=bias, physics_engine=physics_engine)
        self.hh = AnalogLinear(hidden_size, 4 * hidden_size, bias=bias, physics_engine=physics_engine)
    def forward(self, input, state):
        hx, cx = state
        gates = self.ih(input) + self.hh(hx)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        cy = (torch.sigmoid(forgetgate) * cx) + (torch.sigmoid(ingate) * torch.tanh(cellgate))
        hy = torch.sigmoid(outgate) * torch.tanh(cy)
        return hy, cy
def remap_all_weights(model):
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (AnalogLinear, AnalogConv2d)):
                max_w = module.weight.abs().max()
                if max_w > 1e-9: module.weight.div_(max_w); module.alpha.data = max_w
