import torch
from .injection import FixedGeometryFlow, RADIUS, HALF_WIDTH

class SmoothCartesianFlow(FixedGeometryFlow):
    """Smooth Cartesian outputs; velocity BCs are imposed by boundary loss."""
    def forward(self, xy, re):
        p = self.geometry_parameters(re)
        raw = self.net(self.features(xy, p))
        outlet = torch.stack((torch.full_like(re.reshape(-1),4*RADIUS),
                              torch.zeros_like(re.reshape(-1))),dim=1)
        gauge = self.net(self.features(outlet,p))[:,2:3]
        pressure = (4*torch.pi*RADIUS/HALF_WIDTH**2)*(raw[:,2:3]-gauge)
        return torch.cat((raw[:,:2],pressure),dim=1)
