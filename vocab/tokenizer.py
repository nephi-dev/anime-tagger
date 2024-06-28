import contextlib

from constants import BASE_DIR
from torch import Tensor


def load_vocab() -> dict:
    with open(BASE_DIR / "vocab" / "vocab.txt", "r") as file:
        return {word: index for index, word in enumerate(file.read().splitlines())}


def tags_to_tensor(tensor: Tensor, tags: list[str], vocab: dict) -> Tensor:
    _tensor = tensor.clone()
    for tag in tags:
        with contextlib.suppress(KeyError):
            _tensor[vocab[tag]] = 1
    return _tensor


def tensor_to_tags(tensor: Tensor, vocab: dict, threshold: float = 0.5) -> list[str]:
    return [word for word, index in vocab.items() if tensor[index] > threshold]
