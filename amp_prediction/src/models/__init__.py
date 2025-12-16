"""Neural network models for AMP prediction."""

from .cnn import CNN1DAMPClassifier
from .lstm import AMPBilstmClassifier, AMP_BiRNN
from .gru import GRUClassifier
from .hybrid import CNN_BiLSTM_Classifier
from .transformer import AMPTransformerClassifier
from .logistic import LogisticRegression

__all__ = [
    'CNN1DAMPClassifier',
    'AMPBilstmClassifier',
    'AMP_BiRNN',
    'GRUClassifier',
    'CNN_BiLSTM_Classifier',
    'AMPTransformerClassifier',
    'LogisticRegression'
]