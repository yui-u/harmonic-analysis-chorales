import torch
from torch.utils.data import Dataset

from core.common.constants import META_DATA, PAD_INDEX


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class CustomCollate:
    @staticmethod
    def collate(instances):
        batch = {}
        for instance in instances:
            for k, v in instance.items():
                if k not in batch:
                    batch[k] = []
                batch[k].append(v)

        for k, instances in batch.items():
            if 'midi' in k:
                num_parts = [instance.to_tensor().size(0) for instance in instances]
                max_num_parts = max(num_parts)
                if min(num_parts) < max_num_parts:
                    stack_tensors = []
                    for instance in instances:
                        instance_tensor = instance.to_tensor()
                        if instance_tensor.size(0) < max_num_parts:
                            new_size = list(instance_tensor.size())
                            new_size[0] = max_num_parts - instance_tensor.size(0)
                            pad_tensor = torch.ones(torch.Size(new_size), device=instance_tensor.device).long() * PAD_INDEX
                            instance_tensor = torch.cat([instance_tensor, pad_tensor], dim=0)
                        stack_tensors.append(instance_tensor)
                    batch[k] = torch.stack(stack_tensors, dim=0)
                else:
                    batch[k] = torch.stack([instance.to_tensor() for instance in instances], dim=0)
                batch[k] = batch[k].transpose(1, 2)  # (B, L, P) or (B, L, P, O)
            elif k != META_DATA:
                batch[k] = torch.stack([instance.to_tensor() for instance in instances], dim=0)
        return batch
