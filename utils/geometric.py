
import torch
from torch import Tensor

def lineseg_dists(p: Tensor, a: Tensor, b: Tensor) -> Tensor:
    """Cartesian distance from point to line segment in torch.
    Limited to 2 dimensions.

    Adapted from:
    https://stackoverflow.com/a/54442561/11208892

    Args:
        - p: np.array of single point, shape (2,) or 2D array, shape (x, 2)
        - a: np.array of shape (x, 2)
        - b: np.array of shape (x, 2)
    """
    # normalized tangent vectors
    d_ba = b - a
    d = torch.divide(d_ba, (torch.hypot(d_ba[:, 0], d_ba[:, 1])
                            .reshape(-1, 1)))

    # signed parallel distance components
    # rowwise dot products of 2D vectors
    s = torch.multiply(a - p, d).sum(axis=1)
    t = torch.multiply(p - b, d).sum(axis=1)

    # clamped parallel distance
    h = torch.max(torch.stack([s, t, torch.zeros_like(s)]), dim=0)

    # perpendicular distance component
    # rowwise cross products of 2D vectors  
    d_pa = p - a
    c = d_pa[:, 0] * d[:, 1] - d_pa[:, 1] * d[:, 0]

    return torch.hypot(h, c)
