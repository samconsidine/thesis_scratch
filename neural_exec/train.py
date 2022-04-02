from model import MPNNModel
from data import gen_prims_data_instance
from torch_geometric.loader import DataLoader


model = MPNNModel(in_dim=1, emb_dim=1, edge_dim=1, out_dim=1)

data_list = [gen_prims_data_instance(12, 2) for _ in range(64)]
loader = DataLoader(data_list, batch_size=1)

num_epochs = 12


def train_step(model, data):
    preds = model(data)
    print(preds)
    breakpoint()

for epoch in range(num_epochs):
    for data in loader:
        model = train_step(model, data)

