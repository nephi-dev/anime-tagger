from constants import BASE_DIR
from dataset import AnimeDataset
from model import MultiTagger
from polars import DataFrame
from rich.progress import Progress, TextColumn, TimeElapsedColumn
from torch import nn, optim
from torch.utils.data import DataLoader
from vocab.tokenizer import load_vocab


def get_latest_trained_model():
    final_model_path = BASE_DIR / "model.pth"
    if final_model_path.exists():
        return final_model_path
    latest_model = None
    for model in BASE_DIR.glob("model_*.pth"):
        model_epoch = int(model.stem.split("_")[1])
        if latest_model is None or model_epoch > int(latest_model.stem.split("_")[1]):
            latest_model = model
    return latest_model


def mount_model(vocab: dict):
    latest_model = get_latest_trained_model()
    if latest_model is None:
        return MultiTagger(len(vocab))
    return MultiTagger.load(latest_model, len(vocab))


def save_state(
    model: MultiTagger, epochs_losses: list[float], epoch: int | None = None
):
    epoch_str = f"_{epoch}" if epoch is not None else ""
    model.save(BASE_DIR / f"model{epoch_str}.pth")

    df = DataFrame({"loss": epochs_losses})
    df.write_csv(BASE_DIR / "losses.csv")


def train():
    vocab = load_vocab()
    dataset = AnimeDataset(vocab)
    batch_size = 128
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Dataset size: {len(dataset)}")

    epochs = 100
    save_every_epoch = 5
    model = mount_model(vocab)
    print(f"Model({model.int_parameters} params) with {len(vocab)} tags")
    model.train()
    model.cuda()

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    progress = Progress(
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        TextColumn("[bold blue]Loss: {task.fields[loss]:.8f}"),
        TextColumn("[bold green]{task.completed}/{task.total}"),
    )
    epoch_task = progress.add_task("Epoch", total=epochs, loss=0.0)
    batch_task = progress.add_task("Batch", total=len(loader), loss=0.0)
    progress.start()
    epochs_losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for x, y in loader:
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()

            y_pred = model(x)
            loss = criterion(y_pred, y)

            loss.backward()
            optimizer.step()
            batch_loss = loss.item()
            epoch_loss += batch_loss
            progress.update(batch_task, advance=1, loss=batch_loss)
        epochs_losses.append(epoch_loss)
        progress.update(epoch_task, advance=1, loss=epoch_loss)
        progress.reset(batch_task)
        if epoch % save_every_epoch == 0 and epoch != 0:
            save_state(model, epoch_loss, epoch)
    progress.stop()

    save_state(model, epoch_loss)


if __name__ == "__main__":
    train()
