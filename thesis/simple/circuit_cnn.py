import torch
import torch.nn as nn

""""
Complete

pyc(t) = v(t)[>0]
[fast] dv(t)/dt = (1+d(t)[>0]) * (-v(t) + w_FF * cRF(t) + baseline_soma(t) + d(t)[>threshold] + w_pyc * pyc(t)) / (w_PV * pv(t))
[mid] da(t)/dt = -a(t) + pyc(t)
[slow] dd(t)/dt = -d(t) + w_FB * ecRF(t) + baseline_apical(t) - w_SST * sst(t)

[faster]
pv(t) = [w_pvLAT * pyc(t) - w_pv_pv * pv(t)][>0]
vip(t) = [w_vipFB * ecRF(t)][>0]
sst(t) = [w_sstLAT * pyc(t) - w_VIP * vip(t)][>0]

Simple (MLPs + sparsity)
pyc(t) = v(t)[>0]
[fast] dv(t)/dt = -v(t) + (1+d(t)[>0]) * (w_FF * cRF(t) + baseline_soma(t) + d(t)[>threshold] + w_pyc * pyc(t) - w_PV * pv(t))
[mid] da(t)/dt = -a(t) + pyc(t)
[slow] dd(t)/dt = -d(t) + w_FB * ecRF(t) + baseline_apical(t)

[faster]
pv(t) = [w_pvLAT * pyc(t) - w_pv_pv * pv(t)][>0]
"""
class PyC(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PyC, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.model(x)

class PV(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PV, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.model(x)

class CircuitLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CircuitLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pyc = PyC(input_dim, output_dim)
        self.pv = PV(input_dim, output_dim)


    def forward(self, x):
        pyc_out = self.pyc(x)
        pv_out = self.pv(x)

        return pyc_out, pv_out
# Pretrain with predictive learning across saccades JEPA on STL-10 (use seqJEPA approach)

# Alternatively global-local views approach
# Global views downsampled to same res as local views 32x32
# Local views full res 32x32