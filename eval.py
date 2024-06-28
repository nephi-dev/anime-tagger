from pathlib import Path

from constants import BASE_DIR
from dataset import SquarePad
from model import MultiTagger
from PIL import Image
from rich import print
from torchvision.transforms import Compose, Resize, ToTensor
from vocab.tokenizer import load_vocab, tensor_to_tags

MAIN_TRANSFORM = Compose(
    [
        SquarePad(),
        Resize((224, 224)),
        ToTensor(),
    ]
)


def tag_wrong(tag: str):
    return f"[bold red]{tag}[/bold red]"


def tag_right(tag: str):
    return f"[bold green]{tag}[/bold green]"


def validate_tags(tags: list[str], predicted_tags: list[str]):
    res = []
    for tag in tags:
        if tag in predicted_tags:
            res.append(tag_right(tag))
            continue
        res.append(tag_wrong(tag))
    return res


def eval(image_path: Path):
    vocab = load_vocab()
    model = MultiTagger.load(BASE_DIR / "model.pth", len(vocab))
    model = model.eval().cuda().half()
    print(f"Model params: {model.int_parameters}")

    image = Image.open(image_path)
    if image.mode == "RGBA":
        image = image.convert("RGB")
    image = MAIN_TRANSFORM(image)
    image = image.unsqueeze(0).cuda().half()

    tags = model(image)
    tags = tags.squeeze(0).detach().cpu()

    tags = tensor_to_tags(tags, vocab, 0.3)
    return tags


if __name__ == "__main__":
    image_path = Path(input("Enter the path to the image: "))
    try:
        tags = image_path.with_suffix(".txt").read_text().split(", ")
    except FileNotFoundError:
        tags = []

    predicted_tags = eval(image_path)
    print(f"Tags: {', '.join(validate_tags(tags, predicted_tags))}\n")
    print(f"Predicted Tags: {', '.join(validate_tags(predicted_tags, tags))}")
