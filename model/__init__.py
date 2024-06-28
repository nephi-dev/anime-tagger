from torch import Tensor, load, nn, save

# Refs
# https://www.kaggle.com/code/altairfarooque/multi-label-image-classification-cv-3-0
# https://github.com/KichangKim/DeepDanbooru


class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        mid_channels: int | None = None,
        stride: int = 1,
    ):
        super(Bottleneck, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        if mid_channels is None:
            mid_channels = in_channels // 4
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        out = self.bn1(x)
        out = self.relu(out)
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        return out + shortcut


class ResNet(nn.Module):
    def __init__(
        self,
        sizes: list[int] | None = None,
        blocks: list[int] | None = None,
        num_classes: int = 1,
    ):
        if sizes is None:
            sizes = [64, 128, 256, 512]
        if blocks is None:
            blocks = [3, 4, 6, 3]
        super(ResNet, self).__init__()
        self.root = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7, stride=2, padding=3, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        )

        layers = []
        last_size = 8
        for i, (size, blocks_count) in enumerate(zip(sizes, blocks)):
            if i > 0:
                layers.append(Bottleneck(last_size, size, stride=2))
            else:
                layers.append(Bottleneck(last_size, size))
            last_size = size
            layers.append(
                nn.Sequential(*[Bottleneck(size) for _ in range(blocks_count - 1)])
            )
        self.body = nn.Sequential(*layers)

        self.head = nn.Sequential(
            nn.Conv2d(sizes[-1], num_classes, kernel_size=1, bias=False)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.root(x)
        x = self.body(x)
        x = self.head(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        return x.view(x.size(0), -1)


class MultiTagger(nn.Module):
    def __init__(self, num_tags: int):
        super(MultiTagger, self).__init__()
        self.backbone = ResNet(
            sizes=[128, 256, 512, 512, 512, 1024],
            blocks=[2, 7, 40, 16, 16, 6],
            num_classes=num_tags,
        )
        self.out = nn.Sequential(nn.Sigmoid())

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        x = self.out(x)
        return x

    def save(self, path: str):
        save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, num_tags: int):
        model = cls(num_tags)
        model.load_state_dict(load(path))
        return model

    @property
    def int_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
