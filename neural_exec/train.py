from model import make_mpnn
from data import gen_prims_data_instance
from torch_geometric.loader import DataLoader


model = make_mpnn(12, 2)

data_list = [gen_prims_data_instance(12, 2) for _ in range(64)]
loader = DataLoader(data_list, batch_size=32)
breakpoint()

model(loader)

