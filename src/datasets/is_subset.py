from torch.utils.data import Subset


class IS_Subset(Subset):
    """
    Defines dataset with importance sampling weight.
    """

    def __init__(self, dataset, indices, IS_weight) -> None:
        super().__init__(dataset, indices)
        self.weight = IS_weight

    def __getitem__(self, idx):
        if isinstance(idx, list):
            index = [self.indices[i] for i in idx]
            weight = [self.weight[i] for i in idx]
        else:
            index = self.indices[idx]
            weight = self.weight[idx]

        return super().__getitem__(idx) + (weight, index)

    def __len__(self):
        return super().__len__()
