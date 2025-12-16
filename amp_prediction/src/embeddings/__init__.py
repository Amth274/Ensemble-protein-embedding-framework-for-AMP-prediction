"""Embedding generation modules for protein sequences."""

from .esm_embeddings import ESMEmbedding, ESMSequenceEmbedding, ESMAminoAcidEmbedding

__all__ = [
    'ESMEmbedding',
    'ESMSequenceEmbedding',
    'ESMAminoAcidEmbedding'
]