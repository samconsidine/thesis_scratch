
def batch_generator(batch_size: int, n_nodes: int, n_dims: int):

    def _cat_into_batch(instances: List[Data], attr: str) -> Tensor:
        return torch.cat([getattr(data, attr) for data in instances])

    def _cat_into_batch_w_idx(instances: List[Data], attr: str, n_nodes: int) -> Tensor:
        return torch.cat([getattr(data, attr) + n_nodes*i for i, data in enumerate(instances)])


    generators = [gen_prims_data_instance(n_nodes, n_dims) for _ in range(batch_size)]

    for instances in zip(*generators):
        batch_x = _cat_into_batch(instances, 'x')
        batch_x_prev = _cat_into_batch(instances, 'x_prev')
        batch_edge_weights = _cat_into_batch(instances, 'edge_weights')
        batch_edge_index = _cat_into_batch_w_idx(instances, 'edge_index', n_nodes)
        batch_predecessor = _cat_into_batch_w_idx(instances, 'predecessor', n_nodes)

        yield Data(
            x=batch_x, 
            y=batch_x_prev,
            edge_weights=batch_edge_weights, 
            edge_index=batch_edge_index,
            predecessor_index=batch_predecessor
        )

