import torch
from torch import nn as nn
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

PRE_TRAINED_MODEL = "LaBSE"


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class MultilingualSentenceEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SentenceTransformer(PRE_TRAINED_MODEL)

    def forward(self, inputs):
        outputs = self.model(inputs)
        sentence_embedding = outputs.sentence_embedding
        return sentence_embedding


class UNSDataset(Dataset):
    def __init__(self, df):
        self.uns_tokenizer = SentenceTransformer(PRE_TRAINED_MODEL).tokenizer
        self.texts = df["title"].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = self.prepare_uns_input(self.texts[item])
        return inputs

    def prepare_uns_input(self, text):
        inputs = self.uns_tokenizer.encode_plus(
            text,
            return_tensors=None,
            add_special_tokens=True,
        )
        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype=torch.long)
        return inputs
