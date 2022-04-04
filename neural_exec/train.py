import torch 
from itertools import chain
from tqdm import tqdm

from model import MPNN, Encoder, MSTDecoder, PredecessorDecoder
from preprocessing import preprocess_mst
from prims import generate_prims_dataset

# Questions: 
# 1. What is the latent dimension?
# 2. Why mst_logts > 0? No constraint on number of nodes that can be selected?
# 3. (When) Is the latent updated?
# 4. ...

LATENT_DIM = 16
NODE_FEATURES = 1
NUM_NODES = 6

loader = generate_prims_dataset(size=256, num_nodes=NUM_NODES, batch_size=16)
encoder = Encoder(node_feature_dim=1, latent_dim=LATENT_DIM)
processor = MPNN(in_channels=LATENT_DIM, out_channels=LATENT_DIM, bias=True)
mst_decoder = MSTDecoder(LATENT_DIM)

flatten = lambda x: sum(x, start=[])

mst_loss = torch.nn.BCELoss()
optimizer = torch.optim.Adam(flatten(list(x.parameters()) for x in [encoder, processor, mst_decoder]))


def batch_mst_acc(preds, real):
    with torch.no_grad():
        return (in_mst == real).float().mean()


losses = []
for epoch in range(3000):
    batch_ctr = 0.
    acc_avg = 0.
    # Train
    for data in (loader):
        graph_size = data.graph_size[0]
        h = torch.zeros((data.num_nodes, LATENT_DIM))
        loss = 0.
        for step in range(graph_size - 1):
            prev_tree = data.x[:, step:(step+1)]  # Keep dims of slice
            current_tree = data.y[:, step:(step+1)]

            encoded = encoder(prev_tree, h)
            h = processor(x=encoded, edge_attr=data.edge_weights,
                          edge_index=data.edge_index, hidden=h)
            mst_logits = mst_decoder(encoded, h)

            loss += mst_loss(mst_logits, current_tree)

            in_mst = (mst_logits > 0.5).float()  # Why > 0?
            acc_avg += batch_mst_acc(in_mst, current_tree)
            batch_ctr += 1

        loss /= data.graph_size[0] - 1
        print(f":Loss: {loss.item()}")
        losses.append(loss.detach().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} finished. Accuracy: {acc_avg / batch_ctr}.")


import matplotlib.pyplot as plt
plt.plot(losses)
plt.show()
            

