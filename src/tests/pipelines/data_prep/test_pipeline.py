"""
This is a boilerplate test file for pipeline 'data_prep'
generated using Kedro 0.18.4.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""

import pytest
from curriculum_recommendations.pipelines.data_prep.torch_embeddings import (
    MultilingualSentenceEmbedding,
    UNSDataset,
)
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from itertools import combinations


@pytest.fixture
def multilingual_same_sentences():
    texts = pd.Series(
        ["the sky is blue", "le ciel est bleu", "der Himmel ist blau", "el cielo es azul"]
    ).to_frame("title")
    return texts


@pytest.fixture(scope="module")
def embedding_model():
    model = MultilingualSentenceEmbedding()
    return model


@pytest.fixture
def loader(multilingual_same_sentences):
    dataset = UNSDataset(multilingual_same_sentences)
    # Create topics and content dataloaders
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=dataset.uns_tokenizer, padding="longest"),
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    return loader


@pytest.mark.parametrize("device", ["cpu"])
def test_multilingual_embeddings_coherence(embedding_model, loader, device):
    embedding_model.eval()
    preds = []
    for step, inputs in enumerate(loader):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = embedding_model(inputs)
        preds.append(y_preds.to(device).numpy())
    preds = np.concatenate(preds)
    cosine_distances = [cosine(pair[0], pair[1]) for pair in combinations(preds, 2)]
    assert all(dist < 0.1 for dist in cosine_distances)
