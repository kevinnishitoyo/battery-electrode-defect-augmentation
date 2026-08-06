"""Conditional VAE and GAN models used by the multilabel notebooks."""

import torch
from torch import nn


class ConditionalVAE(nn.Module):
    def __init__(self, latent_dim=64, condition_dim=3):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
        )
        flattened_size = 128 * 16 * 16
        self.condition_encoder = nn.Linear(condition_dim, 16)
        self.fc_mu = nn.Linear(flattened_size + 16, latent_dim)
        self.fc_logvar = nn.Linear(flattened_size + 16, latent_dim)
        self.decoder_input = nn.Linear(latent_dim + 16, flattened_size)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def encode(self, images, conditions):
        image_features = self.encoder(images).flatten(start_dim=1)
        condition_features = torch.relu(self.condition_encoder(conditions))
        features = torch.cat([image_features, condition_features], dim=1)
        return self.fc_mu(features), self.fc_logvar(features)

    def reparameterize(self, mu, logvar):
        standard_deviation = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(standard_deviation) * standard_deviation

    def decode(self, latent_vectors, conditions):
        condition_features = torch.relu(self.condition_encoder(conditions))
        features = torch.cat([latent_vectors, condition_features], dim=1)
        decoded = self.decoder_input(features).view(-1, 128, 16, 16)
        return self.decoder(decoded)

    def forward(self, images, conditions):
        mu, logvar = self.encode(images, conditions)
        latent_vectors = self.reparameterize(mu, logvar)
        return self.decode(latent_vectors, conditions), mu, logvar


class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim=100, condition_dim=3):
        super().__init__()
        self.condition_encoder = nn.Linear(condition_dim, 16)
        self.input_layer = nn.Linear(latent_dim + 16, 512 * 4 * 4)
        self.generator = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, latent_vectors, conditions):
        condition_features = torch.relu(self.condition_encoder(conditions))
        features = torch.cat([latent_vectors, condition_features], dim=1)
        features = self.input_layer(features).view(-1, 512, 4, 4)
        return self.generator(features)


class ConditionalDiscriminator(nn.Module):
    def __init__(self, condition_dim=3):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(3 + condition_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 4, 1, 0),
        )

    def forward(self, images, conditions):
        condition_maps = conditions.view(
            conditions.size(0), conditions.size(1), 1, 1
        ).expand(-1, -1, images.size(2), images.size(3))
        inputs = torch.cat([images, condition_maps], dim=1)
        return self.discriminator(inputs).view(-1)
