import torch

class Discriminator(torch.nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [
                torch.nn.Conv2d(in_filters, in_filters, 3, stride=1, padding=1),
                torch.nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1),
                torch.nn.Conv2d(out_filters, out_filters, 3, stride=1, padding=1),
            ]
            if normalization:
                layers.append(torch.nn.InstanceNorm2d(out_filters))
            layers.append(torch.nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = torch.nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            torch.nn.ZeroPad2d((1, 0, 1, 0)),
            torch.nn.Conv2d(512, 1, 4, padding=1, bias=False),
            torch.nn.Tanh(),
        )

    def forward(self, condition, img):
        # Concatenate image and condition image by channels to produce input

        img_input = torch.cat((img, condition), 1)

        output = self.model(img_input)

        return output