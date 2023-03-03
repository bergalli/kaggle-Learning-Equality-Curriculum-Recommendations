"""
This is a boilerplate pipeline 'data_prep'
generated using Kedro 0.18.4
"""
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from tqdm import tqdm
import numpy as np

from .torch_embeddings import MultilingualSentenceEmbedding, UNSDataset


def clean_topics(
    content,
    correlations,
    sample_submission,
    topics,
):
    topics["title"].fillna("", inplace=True)
    content["title"].fillna("", inplace=True)
    print(f"topics.shape: {topics.shape}")
    print(f"content.shape: {content.shape}")
    return False


def read_data(topics, content, sample_submission):
    # Merge topics with sample submission to only infer test topics
    # topics = topics.merge(sample_submission, how="inner", left_on="id", right_on="topic_id")
    # Fillna titles
    topics["title"].fillna("", inplace=True)
    content["title"].fillna("", inplace=True)
    # Sort by title length to make inference faster
    topics["length"] = topics["title"].apply(lambda x: len(x))
    content["length"] = content["title"].apply(lambda x: len(x))
    topics.sort_values("length", inplace=True)
    content.sort_values("length", inplace=True)
    # Drop cols
    topics.drop(
        [
            "description",
            "channel",
            "category",
            "level",
            "parent",
            "has_content",
            "length",
            # "topic_id",
            # "content_ids",
        ],
        axis=1,
        inplace=True,
    )
    content.drop(
        ["description", "kind", "text", "copyright_holder", "license", "length"],
        axis=1,
        inplace=True,
    )
    # Reset index
    topics.reset_index(drop=True, inplace=True)
    content.reset_index(drop=True, inplace=True)
    print(" ")
    print("-" * 50)
    print(f"topics.shape: {topics.shape}")
    print(f"content.shape: {content.shape}")
    return topics, content


def get_embeddings(tmp_topics, tmp_content, device="cpu"):
    # Create topics dataset
    topics_dataset = UNSDataset(tmp_topics)
    # Create content dataset
    content_dataset = UNSDataset(tmp_content)
    # Create topics and content dataloaders
    topics_loader = DataLoader(
        topics_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(
            tokenizer=topics_dataset.uns_tokenizer, padding="longest"
        ),
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    content_loader = DataLoader(
        content_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(
            tokenizer=content_dataset.uns_tokenizer, padding="longest"
        ),
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    def _get_embeddings(loader, model, device):
        model.eval()
        preds = []
        for step, inputs in enumerate(tqdm(loader)):
            for k, v in inputs.items():
                inputs[k] = v.to(device)
            with torch.no_grad():
                y_preds = model(inputs)
            preds.append(y_preds.to(device).numpy())
        preds = np.concatenate(preds)
        return preds

    # Create unsupervised model to extract embeddings
    model = MultilingualSentenceEmbedding()
    model.to(device)
    # Predict topics
    topics_preds = _get_embeddings(topics_loader, model, device)
    content_preds = _get_embeddings(content_loader, model, device)

    return topics_preds, content_preds
