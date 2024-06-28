from asyncio import gather, run
from io import BytesIO
from pathlib import Path

import numpy as np
import PIL
from aiofiles import open as aopen
from constants import BASE_DIR
from rich.progress import Progress, TextColumn, TimeElapsedColumn
from torch import Tensor, load, save, zeros
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.transforms import functional as F
from vocab.tokenizer import tags_to_tensor

PIL.LOAD_TRUNCATED_IMAGES = True


class SquarePad:
    def __call__(self, image: PIL.Image.Image) -> Tensor:
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, "constant")


MAIN_TRANSFORM = Compose(
    [
        SquarePad(),
        Resize((224, 224)),
        ToTensor(),
    ]
)


def save_cache(img_id: int, image: Tensor, text: Tensor):
    cache_dir = BASE_DIR / "cache"
    img_path = cache_dir / f"{img_id}_img.pt"
    text_path = cache_dir / f"{img_id}_text.pt"
    save(image, img_path)
    save(text, text_path)


def load_cache(img_id: int) -> tuple[Tensor, Tensor]:
    cache_dir = BASE_DIR / "cache"
    img_path = cache_dir / f"{img_id}_img.pt"
    text_path = cache_dir / f"{img_id}_text.pt"
    return load(img_path), load(text_path)


def get_cached_images() -> list[int]:
    cache_dir = BASE_DIR / "cache"
    all_files = [path for path in cache_dir.iterdir() if path.suffix == ".pt"]
    return list({int(path.stem.split("_")[0]) for path in all_files})


async def load_image_data(
    image_path: Path,
    vocab: dict,
    progress: Progress,
    task: int,
    zeroed_tensor: Tensor,
    cached_images: list[int],
) -> tuple[Tensor, Tensor]:
    async def open_image(image_path: Path) -> Tensor:
        try:
            async with aopen(image_path, "rb") as image_file:
                image = PIL.Image.open(BytesIO(await image_file.read()))
                if image.mode != "RGB":
                    image = image.convert("RGB")
                return MAIN_TRANSFORM(image)
        except PIL.UnidentifiedImageError:
            return None

    async def open_tags(tags_path: Path, vocab: dict) -> Tensor:
        async with aopen(tags_path, "r", encoding="utf-8") as tags:
            return tags_to_tensor(zeroed_tensor, (await tags.read()).split(", "), vocab)

    img_id = int(image_path.stem)
    if img_id in cached_images:
        res = load_cache(img_id)
    else:
        res = await gather(
            open_image(image_path), open_tags(image_path.with_suffix(".txt"), vocab)
        )
        save_cache(img_id, res[0], res[1])
    progress.update(task, advance=1)
    return res


async def load_all_images(vocab: dict):
    zeroed_tensor = zeros(len(vocab))
    progress = Progress(
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        TextColumn("[bold blue]{task.completed}/{task.total}"),
    )
    total_images = len(list((BASE_DIR / "data").glob("*.txt")))
    cached_images = get_cached_images()
    task = progress.add_task("Loading images", total=total_images)
    progress.start()
    results: list[tuple[Tensor, Tensor]] = []
    for path in (BASE_DIR / "data").iterdir():
        if path.suffix == ".txt":
            continue
        results.append(
            await load_image_data(
                path, vocab, progress, task, zeroed_tensor, cached_images
            )
        )
    progress.stop()

    return results


class AnimeDataset(Dataset):
    def __init__(self, vocab: dict):
        self.vocab = vocab
        self.data: list[Path] = []
        self.labels: list[Tensor] = []
        for img_data, tags_data in run(load_all_images(vocab)):
            if img_data is None:
                continue
            self.data.append(img_data)
            self.labels.append(tags_data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.data[index], self.labels[index]
