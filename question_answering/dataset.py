import torch
import random
from typing import List, Dict, Set
from torch.utils.data import Dataset
from transformers import AutoTokenizer
class MultipleChoiceDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        tokenizer: AutoTokenizer,
        negative_sample: int = 5
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.negative_sample = 5
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    def build_negative_sample(self, features: List[Dict]) ->List[Dict]:
        min_len = min([len(feature['input_ids']) for feature in features])
        sample_size = min(min_len, self.negative_sample)
        labels = [feature['label'] for feature in features]
        
        batch_samples = []
        for i, feature in enumerate(features):
            label_id = labels[i]
            num_removed = len(feature['input_ids']) - sample_size
            reserved_ids = [i for i in range(len(feature['input_ids'])) if i != label_id]
            removed_ids = random.sample(reserved_ids, num_removed)
            for removed_id in removed_ids:
                if removed_id < labels[i]:
                    label_id -= 1
                reserved_ids.remove(removed_id)
            reserved_ids.append(labels[i])
            reserved_ids.sort()

            batch_sample = {col: [] for col in feature.keys() if col != 'label'}
            for reserved_id in reserved_ids:
                for col in batch_sample.keys():
                    batch_sample[col].append(feature[col][reserved_id])
            labels[i] = label_id
            batch_samples.append(batch_sample)
        
        return batch_samples, labels

    def train_collate_fn(self, features: List[Dict]) -> Set:
        batch_samples, labels = self.build_negative_sample(features)

        batch_size = len(batch_samples)
        num_choices = len(batch_samples[0]["input_ids"])
        flattened_samples = [
            [{k: v[i] for k, v in batch_sample.items() if k != 'label'} for i in range(num_choices)] for batch_sample in batch_samples
        ]
        flattened_samples = sum(flattened_samples, [])
        
        batch_samples = self.tokenizer.pad(
            flattened_samples,
            padding='longest',
            return_tensors="pt",
        )

        # Un-flatten
        batch_samples = {k: v.view(batch_size, num_choices, -1) for k, v in batch_samples.items()}
        # Add back labels
        batch_samples["labels"] = torch.tensor(labels, dtype=torch.int64)

        return batch_samples

    def eval_collate_fn(self, features: List[Dict]) -> Set:
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items() if k != 'label'} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding='longest',
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        if 'label' in features[0].keys():
            labels = [feature['label'] for feature in features]
            batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch

class QuestionAnswerDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        tokenizer: AutoTokenizer,
    ):
        self.data = data
        self.tokenizer = tokenizer
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    def build_negative_sample(self, features: List[Dict]) ->List[Dict]:
        min_len = min([len(feature['input_ids']) for feature in features])
        sample_size = min(min_len, self.negative_sample)
        labels = [feature['label'] for feature in features]
        
        batch_samples = []
        for i, feature in enumerate(features):
            label_id = labels[i]
            num_removed = len(feature['input_ids']) - sample_size
            reserved_ids = [i for i in range(len(feature['input_ids'])) if i != label_id]
            removed_ids = random.sample(reserved_ids, num_removed)
            for removed_id in removed_ids:
                if removed_id < labels[i]:
                    label_id -= 1
                reserved_ids.remove(removed_id)
            reserved_ids.append(labels[i])
            reserved_ids.sort()

            batch_sample = {col: [] for col in feature.keys() if col != 'label'}
            for reserved_id in reserved_ids:
                for col in batch_sample.keys():
                    batch_sample[col].append(feature[col][reserved_id])
            labels[i] = label_id
            batch_samples.append(batch_sample)
        
        return batch_samples, labels

    def train_collate_fn(self, features: List[Dict]) -> Set:
        batch_samples, labels = self.build_negative_sample(features)

        batch_size = len(batch_samples)
        num_choices = len(batch_samples[0]["input_ids"])
        flattened_samples = [
            [{k: v[i] for k, v in batch_sample.items() if k != 'label'} for i in range(num_choices)] for batch_sample in batch_samples
        ]
        flattened_samples = sum(flattened_samples, [])
        
        batch_samples = self.tokenizer.pad(
            flattened_samples,
            padding='longest',
            return_tensors="pt",
        )

        # Un-flatten
        batch_samples = {k: v.view(batch_size, num_choices, -1) for k, v in batch_samples.items()}
        # Add back labels
        batch_samples["labels"] = torch.tensor(labels, dtype=torch.int64)

        return batch_samples

    def eval_collate_fn(self, features: List[Dict]) -> Set:
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items() if k != 'label'} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding='longest',
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        if 'label' in features[0].keys():
            labels = [feature['label'] for feature in features]
            batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch

