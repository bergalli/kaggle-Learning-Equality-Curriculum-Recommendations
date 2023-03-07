# =================================================================================
# Libraries
# =========================================================================================
import os
import gc
import time
import math
import random
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.checkpoint import checkpoint
import tokenizers
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_cosine_schedule_with_warmup, DataCollatorWithPadding
# import cupy as cp
# from cuml.metrics import pairwise_distances
# from cuml.neighbors import NearestNeighbors

TOKENIZERS_PARALLELISM = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CFG1:
    uns_model = "/kaggle/input/lecr-ensemble-data1/sentence-transformers-all-mpnet-base-v2"
    sup_model = "/kaggle/input/lecr-ensemble-data1/xlm-roberta-base"
    uns_tokenizer = AutoTokenizer.from_pretrained(uns_model + '/tokenizer')
    sup_tokenizer = AutoTokenizer.from_pretrained(sup_model + '/tokenizer')
    pooling = "mean"
    gradient_checkpointing = False
    add_with_best_prob = True


class CFG2:
    uns_model = "/kaggle/input/lecr-ensemble-data1/sentence-transformers-all-MiniLM-L6-v2"
    sup_model = "/kaggle/input/lecr-ensemble-data1/xlm-roberta-base"
    uns_tokenizer = AutoTokenizer.from_pretrained(uns_model + '/tokenizer')
    sup_tokenizer = AutoTokenizer.from_pretrained(sup_model + '/tokenizer')
    pooling = "mean"
    gradient_checkpointing = False
    add_with_best_prob = True


class CFG3:
    uns_model = "/kaggle/input/stsbrobertabasev2exp2/stsb-roberta-base-v2-exp1"
    sup_model = "/kaggle/input/lecr-ensemble-data1/xlm-roberta-base"
    uns_tokenizer = AutoTokenizer.from_pretrained(uns_model + '/tokenizer')
    sup_tokenizer = AutoTokenizer.from_pretrained(sup_model + '/tokenizer')
    pooling = "mean"
    gradient_checkpointing = False
    add_with_best_prob = True


class CFG4:
    uns_model = "/kaggle/input/paraphrasedistilrobertabasev1exp2/paraphrase-distilroberta-base-v1-exp2"
    sup_model = "/kaggle/input/lecr-ensemble-data1/xlm-roberta-base"
    uns_tokenizer = AutoTokenizer.from_pretrained(uns_model + '/tokenizer')
    sup_tokenizer = AutoTokenizer.from_pretrained(sup_model + '/tokenizer')
    pooling = "mean"
    gradient_checkpointing = False
    add_with_best_prob = False


class CFG5:
    uns_model = "/kaggle/input/paraphrasemultilingualmpnetbasev4-2/paraphrase-multilingual-mpnet-base-v4"
    sup_model = "/kaggle/input/lecr-ensemble-data1/xlm-roberta-base"
    uns_tokenizer = AutoTokenizer.from_pretrained(uns_model + '/tokenizer')
    sup_tokenizer = AutoTokenizer.from_pretrained(sup_model + '/tokenizer')
    pooling = "mean"
    gradient_checkpointing = False
    add_with_best_prob = False


CFG_list = [CFG1, CFG2, CFG3]


def read_data():
    topics = pd.read_csv('/kaggle/input/learning-equality-curriculum-recommendations/topics.csv')
    content = pd.read_csv('/kaggle/input/learning-equality-curriculum-recommendations/content.csv')
    sample_submission = pd.read_csv('/kaggle/input/learning-equality-curriculum-recommendations/sample_submission.csv')
    # Merge topics with sample submission to only infer test topics
    topics = topics.merge(sample_submission, how='inner', left_on='id', right_on='topic_id')
    # Fillna titles
    topics['title'].fillna("", inplace=True)
    content['title'].fillna("", inplace=True)
    # Sort by title length to make inference faster
    topics['length'] = topics['title'].apply(lambda x: len(x))
    content['length'] = content['title'].apply(lambda x: len(x))
    topics.sort_values('length', inplace=True)
    content.sort_values('length', inplace=True)
    # Drop cols
    topics.drop(
        ['description', 'channel', 'category', 'level', 'parent', 'has_content', 'length', 'topic_id', 'content_ids'],
        axis=1, inplace=True)
    content.drop(['description', 'kind', 'text', 'copyright_holder', 'license', 'length'], axis=1, inplace=True)
    # Reset index
    topics.reset_index(drop=True, inplace=True)
    content.reset_index(drop=True, inplace=True)
    print(' ')
    print('-' * 50)
    print(f"topics.shape: {topics.shape}")
    print(f"content.shape: {content.shape}")
    return topics, content


def prepare_uns_input(text, cfg):
    inputs = cfg.uns_tokenizer.encode_plus(
        text,
        return_tensors=None,
        add_special_tokens=True,
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


# =========================================================================================
# pooling class
# =========================================================================================
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


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, last_hidden_state, attention_mask):
        w = self.attention(last_hidden_state).float()
        w[attention_mask == 0] = float('-inf')
        w = torch.softmax(w, 1)
        attention_embeddings = torch.sum(w * last_hidden_state, dim=1)
        return attention_embeddings


class MaxPooling(nn.Module):
    def __init__(self):
        super(MaxPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        embeddings = last_hidden_state.clone()
        embeddings[input_mask_expanded == 0] = -1e4
        max_embeddings, _ = torch.max(embeddings, dim=1)
        return max_embeddings


class MinPooling(nn.Module):
    def __init__(self):
        super(MinPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        embeddings = last_hidden_state.clone()
        embeddings[input_mask_expanded == 0] = 1e-4
        min_embeddings, _ = torch.min(embeddings, dim=1)
        return min_embeddings


class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights=None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None \
            else nn.Parameter(
            torch.tensor([1] * (num_hidden_layers + 1 - layer_start), dtype=torch.float)
        )

    def forward(self, features):
        ft_all_layers = features['all_layer_embeddings']

        all_layer_embedding = torch.stack(ft_all_layers)
        all_layer_embedding = all_layer_embedding[self.layer_start:, :, :, :]

        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor * all_layer_embedding).sum(dim=0) / self.layer_weights.sum()

        features.update({'token_embeddings': weighted_average})
        return features


class ConcatPooling(nn.Module):
    def __init__(self, backbone_config, pooling_config):
        super(ConcatPooling, self, ).__init__()

        self.n_layers = pooling_config.n_layers
        self.output_dim = backbone_config.hidden_size * pooling_config.n_layers

    def forward(self, inputs, backbone_outputs):
        all_hidden_states = get_all_hidden_states(backbone_outputs)

        concatenate_pooling = torch.cat([all_hidden_states[-(i + 1)] for i in range(self.n_layers)], -1)
        concatenate_pooling = concatenate_pooling[:, 0]
        return concatenate_pooling


# =========================================================================================
# Unsupervised dataset
# =========================================================================================
class uns_dataset(Dataset):
    def __init__(self, df, cfg):
        self.cfg = cfg
        self.texts = df['title'].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_uns_input(self.texts[item], self.cfg)
        return inputs


# =========================================================================================
# Prepare input, tokenize
# =========================================================================================
def prepare_sup_input(text, cfg):
    inputs = cfg.sup_tokenizer.encode_plus(
        text,
        return_tensors=None,
        add_special_tokens=True,
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


# =========================================================================================
# Supervised dataset
# =========================================================================================
class sup_dataset(Dataset):
    def __init__(self, df, cfg):
        self.cfg = cfg
        self.texts = df['text'].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_sup_input(self.texts[item], self.cfg)
        return inputs


# =========================================================================================
# Unsupervised model
# =========================================================================================
class uns_model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.config = AutoConfig.from_pretrained(cfg.uns_model + '/config')
        self.model = AutoModel.from_pretrained(cfg.uns_model + '/model', config=self.config)
        self.pool = MeanPooling()

    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        feature = self.pool(last_hidden_state, inputs['attention_mask'])
        return feature

    def forward(self, inputs):
        feature = self.feature(inputs)
        return feature


# =========================================================================================
# Get embeddings
# =========================================================================================
def get_embeddings(loader, model, device):
    model.eval()
    preds = []
    for step, inputs in enumerate(tqdm(loader)):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
        preds.append(y_preds.to('cpu').numpy())
    preds = np.concatenate(preds)
    return preds


# =========================================================================================
# Get the amount of positive classes based on the total
# =========================================================================================
def get_pos_socre(y_true, y_pred):
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    int_true = np.array([len(x[0] & x[1]) / len(x[0]) for x in zip(y_true, y_pred)])
    return round(np.mean(int_true), 5)


# =========================================================================================
# Build our inference set
# =========================================================================================
def build_inference_set(topics, content, cfg):
    # Create lists for training
    topics_ids = []
    content_ids = []
    topics_languages = []
    content_languages = []
    title1 = []
    title2 = []
    # Iterate over each topic
    for k in tqdm(range(len(topics))):
        row = topics.iloc[k]
        topics_id = row['id']
        topics_language = row['language']
        topics_title = row['title']
        predictions = row['predictions'].split(' ')
        for pred in predictions:
            content_title = content.loc[pred, 'title']
            content_language = content.loc[pred, 'language']
            topics_ids.append(topics_id)
            content_ids.append(pred)
            title1.append(topics_title)
            title2.append(content_title)
            topics_languages.append(topics_language)
            content_languages.append(content_language)
    # Build training dataset
    test = pd.DataFrame(
        {'topics_ids': topics_ids,
         'content_ids': content_ids,
         'title1': title1,
         'title2': title2,
         'topic_language': topics_languages,
         'content_language': content_languages,
         }
    )
    # Release memory
    del topics_ids, content_ids, title1, title2, topics_languages, content_languages
    gc.collect()

    return test


# =========================================================================================
# Get neighbors
# =========================================================================================

def get_neighbors(tmp_topics, tmp_content, cfg):
    # Create topics dataset
    topics_dataset = uns_dataset(tmp_topics, cfg)
    # Create content dataset
    content_dataset = uns_dataset(tmp_content, cfg)
    # Create topics and content dataloaders
    topics_loader = DataLoader(
        topics_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=cfg.uns_tokenizer, padding='longest'),
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    content_loader = DataLoader(
        content_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=cfg.uns_tokenizer, padding='longest'),
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    # Create unsupervised model to extract embeddings
    model = uns_model(cfg)
    model.to(device)
    # Predict topics
    topics_preds = get_embeddings(topics_loader, model, device)
    content_preds = get_embeddings(content_loader, model, device)
    # Transfer predictions to gpu
    topics_preds_gpu = torch.tensor(topics_preds)
    content_preds_gpu = torch.tensor(content_preds)
    # Release memory
    del topics_dataset, content_dataset, topics_loader, content_loader, topics_preds, content_preds
    gc.collect()
    torch.cuda.empty_cache()
    # KNN model
    print(' ')
    print('Training KNN model...')
    neighbors_model = NearestNeighbors(n_neighbors=1000, metric='cosine')
    neighbors_model.fit(content_preds_gpu)
    indices = neighbors_model.kneighbors(topics_preds_gpu, return_distance=False)
    predictions = []
    for k in range(len(indices)):
        pred = indices[k]
        p = ' '.join([tmp_content.loc[ind, 'id'] for ind in pred.get()])
        predictions.append(p)
    tmp_topics['predictions'] = predictions
    # Release memory
    del topics_preds_gpu, content_preds_gpu, neighbors_model, predictions, indices, model
    gc.collect()
    torch.cuda.empty_cache()
    return tmp_topics, tmp_content


# =========================================================================================
# Process test
# =========================================================================================
def preprocess_test(tmp_test):
    tmp_test['title1'].fillna("Title does not exist", inplace=True)
    tmp_test['title2'].fillna("Title does not exist", inplace=True)
    # Create feature column
    tmp_test['text'] = tmp_test['title1'] + '[SEP]' + tmp_test['title2']
    # Drop titles
    tmp_test.drop(['title1', 'title2'], axis=1, inplace=True)
    # Sort so inference is faster
    tmp_test['length'] = tmp_test['text'].apply(lambda x: len(x))
    tmp_test.sort_values('length', inplace=True)
    tmp_test.drop(['length'], axis=1, inplace=True)
    tmp_test.reset_index(drop=True, inplace=True)
    gc.collect()
    torch.cuda.empty_cache()
    return tmp_test


# =========================================================================================
# Model
# =========================================================================================
class custom_model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.config = AutoConfig.from_pretrained(cfg.sup_model + '/config', output_hidden_states=True)
        self.config.hidden_dropout = 0.0
        self.config.hidden_dropout_prob = 0.0
        self.config.attention_dropout = 0.0
        self.config.attention_probs_dropout_prob = 0.0
        self.model = AutoModel.from_pretrained(cfg.sup_model + '/model', config=self.config)
        # self.pool = MeanPooling()
        if self.cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        if CFG.pooling == 'mean' or CFG.pooling == "ConcatPool":
            self.pool = MeanPooling()
        elif CFG.pooling == 'max':
            self.pool = MaxPooling()
        elif CFG.pooling == 'min':
            self.pool = MinPooling()
        elif CFG.pooling == 'attention':
            self.pool = AttentionPooling(self.config.hidden_size)
        elif CFG.pooling == "WLP":
            self.pool = WeightedLayerPooling(self.config.num_hidden_layers, layer_start=6)

        if CFG.pooling == "ConcatPool":
            self.fc = nn.Linear(self.config.hidden_size * 4, 1)
        else:
            self.fc = nn.Linear(self.config.hidden_size, 1)
        # self.fc = nn.Linear(self.config.hidden_size, 1)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def feature(self, inputs):
        outputs = self.model(**inputs)

        if CFG.pooling == "WLP":
            last_hidden_state = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            tmp = {
                'all_layer_embeddings': last_hidden_state.hidden_states
            }
            feature = self.pool(tmp)['token_embeddings'][:, 0]

        elif CFG.pooling == "ConcatPool":
            last_hidden_state = torch.stack(
                self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask']).hidden_states)

            p1 = self.pool(last_hidden_state[-1], inputs['attention_mask'])
            p2 = self.pool(last_hidden_state[-2], inputs['attention_mask'])
            p3 = self.pool(last_hidden_state[-3], inputs['attention_mask'])
            p4 = self.pool(last_hidden_state[-4], inputs['attention_mask'])

            feature = torch.cat(
                (p1, p2, p3, p4), -1
            )

        else:
            last_hidden_state = outputs.last_hidden_state
            feature = self.pool(last_hidden_state, inputs['attention_mask'])

        # last_hidden_state = outputs.last_hidden_state
        # feature = self.pool(last_hidden_state, inputs['attention_mask'])
        return feature

    def forward(self, inputs):
        feature = self.feature(inputs)
        output = self.fc(feature)
        return output


# =========================================================================================
# Inference function loop
# =========================================================================================
def inference_fn(test_loader, model, device):
    preds = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total=len(test_loader))
    for inputs in tk0:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
        preds.append(y_preds.sigmoid().squeeze().to('cpu').numpy().reshape(-1))
    predictions = np.concatenate(preds)
    return predictions


# =========================================================================================
# Inference
# =========================================================================================
def inference(test, cfg, _idx):
    # Create dataset and loader
    test_dataset = sup_dataset(test, cfg)
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=cfg.sup_tokenizer, padding='longest'),
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )
    # Get model
    model = custom_model(cfg)

    # Load weights
    state = torch.load("/kaggle/input/lecr-ensemble-data1/xlm-roberta-base_fold0_42.pth",
                       map_location=torch.device('cpu'))
    model.load_state_dict(state['model'])
    prediction = inference_fn(test_loader, model, device)

    # Release memory
    torch.cuda.empty_cache()
    del test_dataset, test_loader, model, state
    gc.collect()

    # Use threshold
    test['probs'] = prediction
    test['predictions'] = test['probs'].apply(lambda x: int(x > 0.0006))
    test = test.merge(test.groupby("topics_ids", as_index=False)["probs"].max(), on="topics_ids", suffixes=["", "_max"])
    #display(test.head())

    test1 = test[(test['predictions'] == 1) & (test['topic_language'] == test['content_language'])]
    test1 = test1.groupby(['topics_ids'])['content_ids'].unique().reset_index()
    test1['content_ids'] = test1['content_ids'].apply(lambda x: ' '.join(x))
    test1.columns = ['topic_id', 'content_ids']
    #display(test1.head())

    test0 = pd.Series(test['topics_ids'].unique())
    test0 = test0[~test0.isin(test1['topic_id'])]
    test0 = pd.DataFrame({'topic_id': test0.values, 'content_ids': ""})
    if cfg.add_with_best_prob:
        test0 = test0[["topic_id"]].merge(test[test['probs'] == test['probs_max']][["topics_ids", "content_ids"]],
                                          left_on="topic_id", right_on="topics_ids")[['topic_id', "content_ids"]]
    #display(test0.head())
    test_r = pd.concat([test1, test0], axis=0, ignore_index=True)
    test_r.to_csv(f'submission_{_idx + 1}.csv', index=False)

    return test_r

if __name__ == "__main__":
    for _idx, CFG in enumerate(CFG_list):
        # Read data
        tmp_topics, tmp_content = read_data()
        # Run nearest neighbors
        tmp_topics, tmp_content = get_neighbors(tmp_topics, tmp_content, CFG)
        gc.collect()
        torch.cuda.empty_cache()
        # Set id as index for content
        tmp_content.set_index('id', inplace=True)
        # Build training set
        tmp_test = build_inference_set(tmp_topics, tmp_content, CFG)
        # Process test set
        tmp_test = preprocess_test(tmp_test)
        # Inference
        inference(tmp_test, CFG, _idx)
        del tmp_topics, tmp_content, tmp_test
        gc.collect()
        torch.cuda.empty_cache()

    df_test = pd.concat([pd.read_csv(f'submission_{_idx + 1}.csv') for _idx in range(len(CFG_list))])
    df_test.fillna("", inplace=True)
    df_test['content_ids'] = df_test['content_ids'].apply(lambda c: c.split(' '))
    df_test = df_test.explode('content_ids').groupby(['topic_id'])['content_ids'].unique().reset_index()
    df_test['content_ids'] = df_test['content_ids'].apply(lambda c: ' '.join(c))

    df_test.to_csv('submission.csv', index=False)
    df_test.head()