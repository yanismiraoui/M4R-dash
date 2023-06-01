"""
Various model utils
"""

import os, sys
import tempfile
import subprocess
import json
import logging
from itertools import zip_longest
from typing import *

import numpy as np
import pandas as pd
from scipy.special import softmax

import torch
import torch.nn as nn
import skorch

from transformers import (
    AutoModel,
    BertModel,
    BertForMaskedLM,
    BertForNextSentencePrediction,
    BertForSequenceClassification,
    ConvBertForMaskedLM,
    FillMaskPipeline,
    FeatureExtractionPipeline,
    TextClassificationPipeline,
    Pipeline,
    TrainerCallback,
    TrainerControl,
)

from neptune.experiments import Experiment
from neptune.api_exceptions import ChannelsValuesSendBatchError
from transformers.utils.dummy_pt_objects import AutoModelForMaskedLM

import gdown

import data_loader as dl
import featurization as ft
import utils

from transformer_custom import (
    BertForSequenceClassificationMulti,
    BertForThreewayNextSentencePrediction,
    TwoPartBertClassifier,
)


# https://drive.google.com/u/1/uc?id=1VZ1qyNmeYu7mTdDmSH1i00lKIBY26Qoo&export=download
FINETUNED_DUAL_MODEL_ID = "1VZ1qyNmeYu7mTdDmSH1i00lKIBY26Qoo"
FINETUNED_DUAL_MODEL_BASENAME = "tcrbert_lcmv_finetuned_1.0.tar.gz"
FINETUNED_DUAL_MODEL_URL = f"https://drive.google.com/uc?id={FINETUNED_DUAL_MODEL_ID}"
FINETUNED_DUAL_MODEL_MD5 = "e51d8ae58974c2e02d37fd4b51d448ee"
FINETUNED_MODEL_CACHE_DIR = os.path.join(
    os.path.expanduser("~"), ".cache/gdown/tcrbert"
)

def get_transformer_embeddings(
    model_dir: str,
    seqs: Iterable[str],
    seq_pair: Optional[Iterable[str]] = None,
    *,
    layers: List[int] = [-1],
    method: Literal["mean", "max", "attn_mean", "cls", "pool"] = "mean",
    batch_size: int = 256,
    device: int = 0,
) -> np.ndarray:
    """
    Get the embeddings for the given sequences from the given layers
    Layers should be given as negative integers, where -1 indicates the last
    representation, -2 second to last, etc.
    Returns a matrix of num_seqs x (hidden_dim * len(layers))
    Methods:
    - cls:  value of initial CLS token
    - mean: average of sequence length, excluding initial CLS token
    - max:  maximum over sequence length, excluding initial CLS token
    - attn_mean: mean over sequenced weighted by attention, excluding initial CLS token
    - pool: pooling layer
    If multiple layers are given, applies the given method to each layers
    and concatenate across layers
    """
    device = utils.get_device(device)
    seqs = [s if ft.is_whitespaced(s) else ft.insert_whitespace(s) for s in seqs]
    try:
        tok = ft.get_pretrained_bert_tokenizer(model_dir)
    except OSError:
        logging.warning("Could not load saved tokenizer, loading fresh instance")
        tok = ft.get_aa_bert_tokenizer(64)
    model = BertModel.from_pretrained(model_dir, add_pooling_layer=method == "pool").to(
        device
    )

    chunks = dl.chunkify(seqs, batch_size)
    # This defaults to [None] to zip correctly
    chunks_pair = [None]
    if seq_pair is not None:
        assert len(seq_pair) == len(seqs)
        chunks_pair = dl.chunkify(
            [s if ft.is_whitespaced(s) else ft.insert_whitespace(s) for s in seq_pair],
            batch_size,
        )
    # For single input, we get (list of seq, None) items
    # for a duo input, we get (list of seq1, list of seq2)
    chunks_zipped = list(zip_longest(chunks, chunks_pair))
    embeddings = []
    with torch.no_grad():
        for seq_chunk in chunks_zipped:
            encoded = tok(
                *seq_chunk, padding="max_length", max_length=64, return_tensors="pt"
            )
            # manually calculated mask lengths
            # temp = [sum([len(p.split()) for p in pair]) + 3 for pair in zip(*seq_chunk)]
            input_mask = encoded["attention_mask"].numpy()
            encoded = {k: v.to(device) for k, v in encoded.items()}
            # encoded contains input attention mask of (batch, seq_len)
            x = model.forward(
                **encoded, output_hidden_states=True, output_attentions=True
            )
            if method == "pool":
                embeddings.append(x.pooler_output.cpu().numpy().astype(np.float64))
                continue
            # x.hidden_states contains hidden states, num_hidden_layers + 1 (e.g. 13)
            # Each hidden state is (batch, seq_len, hidden_size)
            # x.hidden_states[-1] == x.last_hidden_state
            # x.attentions contains attention, num_hidden_layers
            # Each attention is (batch, attn_heads, seq_len, seq_len)

            for i in range(len(seq_chunk[0])):
                e = []
                for l in layers:
                    # Select the l-th hidden layer for the i-th example
                    h = (
                        x.hidden_states[l][i].cpu().numpy().astype(np.float64)
                    )  # seq_len, hidden
                    # initial 'cls' token
                    if method == "cls":
                        e.append(h[0])
                        continue
                    # Consider rest of sequence
                    if seq_chunk[1] is None:
                        seq_len = len(seq_chunk[0][i].split())  # 'R K D E S' = 5
                    else:
                        seq_len = (
                            len(seq_chunk[0][i].split())
                            + len(seq_chunk[1][i].split())
                            + 1  # For the sep token
                        )
                    seq_hidden = h[1 : 1 + seq_len]  # seq_len * hidden
                    assert len(seq_hidden.shape) == 2
                    if method == "mean":
                        e.append(seq_hidden.mean(axis=0))
                    elif method == "max":
                        e.append(seq_hidden.max(axis=0))
                    elif method == "attn_mean":
                        # (attn_heads, seq_len, seq_len)
                        # columns past seq_len + 2 are all 0
                        # summation over last seq_len dim = 1 (as expected after softmax)
                        attn = x.attentions[l][i, :, :, : seq_len + 2]
                        # print(attn.shape)
                        print(attn.sum(axis=-1))
                        raise NotImplementedError
                    else:
                        raise ValueError(f"Unrecognized method: {method}")
                e = np.hstack(e)
                assert len(e.shape) == 1
                embeddings.append(e)
    if len(embeddings[0].shape) == 1:
        embeddings = np.stack(embeddings)
    else:
        embeddings = np.vstack(embeddings)
    del x
    del model
    torch.cuda.empty_cache()
    return embeddings