
import torch 
import torch.nn as nn 
import torch.nn.functional as F

from .layers import Binary, MLPBlock

ONEOVERSQRT2PI = 1.0 / torch.sqrt(torch.tensor(2 * torch.pi))
LOG2PI = torch.log(torch.tensor(2 * torch.pi))

# Freeze Model Parameters
def freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False
        
# Unfreeze Model Parameters
def unfreeze_module(module):
    for param in module.parameters():
        param.requires_grad = True
        

# Encoder class for transforming Morganfingerprints to latent space vector
class Encoder(nn.Module):
    def __init__(self, key_length, latent_dim):
        super(Encoder, self).__init__()

        self.embed = nn.Linear(key_length, 512)
        self.mlp = MLPBlock(512, latent_dim, 5, act=nn.SiLU())
        
        
    def forward(self, fingerprints):
        x = self.embed(fingerprints)
        # Apply sigmoid to constraint to latent vector to scale features into [0, 1] range
        x = torch.sigmoid(self.mlp(x))
        return x


# Simple decoder model to assist in the training of the AAE
class Decoder(nn.Module):
    def __init__(self, key_length, latent_dim, norm: nn.Module = nn.BatchNorm1d):
        super(Decoder, self).__init__()
        
        # Model uses a rather linear architecture to broadcast from latent space, however it peforms well for the task at hand. 
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            norm(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            norm(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(512, 512),
            norm(512),
            nn.LeakyReLU(0.05, inplace=True),
            nn.Linear(512, key_length, bias=False),
            nn.Tanh(),
            # Apply Binary Mask to output to match the fingerprint bit-vector
            Binary()
        )
        

    def forward(self, z):
        return self.model(z)


# Discriminator model for the AAE
class Discriminator(nn.Module):
    def __init__(self, latent_dim, norm: nn.Module = nn.BatchNorm1d):
        super(Discriminator, self).__init__()

        # The layers are individually defined, rather than using a MLPBlock to allow easier fine-tuning of the layer types
        self.fc1 = nn.Linear(latent_dim, 256)
        self.norm1 = norm(256)
        self.fc2 = nn.Linear(256, 128)
        self.norm2 = norm(128)
        self.fc3 =  nn.Linear(128, 64)
        self.norm3 = norm(64)
        self.fc4 = nn.Linear(64, 32)
        self.norm4 = norm(32)
        self.fc5 = nn.Linear(32, 1)
        

    def forward(self, x):
        # Batchnorm is only applied if the batchsize is greater than 1
        b = x.shape[0]
        x = self.fc1(x)
        if b > 1:
            x = self.norm1(x)
        x = F.silu(x, 0.2)
        x = self.fc2(x)
        if b > 1:
            x = self.norm2(x)
        x = F.silu(x)
        x = self.fc3(x)
        if b > 1:
            x = self.norm3(x)
        x = F.silu(x)
        x = self.fc4(x)
        if b > 1:
            x = self.norm4(x)
        x = F.silu(x)
        return self.fc5(x)
               

# REGRESSION TASK MODELS

# Vanilla Regression Model
class KinaseRegressionModel(nn.Module):
    
    def __init__(self, key_length, latent_dim, activation=nn.Module):
        super().__init__()
            
        # Encoder model for transforming the fingerprint to latent space
        self.encoder = Encoder(key_length, latent_dim)
        # Discriminator model to subsection out layers for the regression task        
        self.discriminator_layer = Discriminator(latent_dim, norm=nn.BatchNorm1d)
        
        # Coupling layers for the encoder/discriminator models
        self.discriminant = MLPBlock(latent_dim, latent_dim // 2, 2, activation)
        self.regressor = MLPBlock(latent_dim, latent_dim // 2, 2, activation)
        
        # Main Regression MLP
        self.final_mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.BatchNorm1d(latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, latent_dim // 4),
            nn.BatchNorm1d(latent_dim // 4),
            nn.ReLU(),
            nn.Linear(latent_dim // 4, 1),
        )
        self.act = activation
        
        
    def forward(self, feats):
        ## cast fingerprints into latent space, followed by encoder coupling layer
        # [B, 1024] -> [B, latent_dim]
        latent_x = self.encoder(feats)
        # [B, latent_dim] -> [B, latent_dim // 2]
        latent_fp = self.act(self.regressor(latent_x))
        ## cast the latent vector through the pretrained disciminator layer fc1
        discrim_x = latent_x
        # [B, latent_dim] -> [B, 256]
        discrim_x = self.discriminator_layer.fc1(discrim_x)
        # discriminator coupling layer 
        # [B, latent_dim] -> [B, latent_dim // 2]
        discrim_x = self.act(self.discriminant(discrim_x))
        
        x_cat = torch.cat([latent_fp, discrim_x], dim=-1)
        x_cat = self.act(self.final_mlp(x_cat))
        return x_cat


# Regression Model with Mixture of Gaussians
class KinaseProbRegressionModel(nn.Module):
    
    def __init__(self, key_length, latent_dim, num_gauss, activation=nn.Module):
        super().__init__()
            
        # Assign input models and freeze their parameters
        self.encoder = Encoder(key_length, latent_dim)
        self.discriminator_layer = Discriminator(latent_dim, norm=nn.BatchNorm1d)
        
        # Coupling layers for the encoder/discriminator models
        self.discriminant = MLPBlock(latent_dim, latent_dim // 2, 2, activation)
        self.regressor = MLPBlock(latent_dim, latent_dim // 2, 2, activation)
        
        self.pi = MLPBlock(latent_dim, num_gauss, 4, activation)
        self.mu = MLPBlock(latent_dim, num_gauss, 4, activation)
        self.sigma = MLPBlock(latent_dim, num_gauss, 4, activation)
        
        self.act = activation
        
    def forward(self, x):
        latent_x = self.encoder(x)
        discrim_x = self.discriminator_layer.fc1(latent_x)
        
        coup_encoder = self.act(self.regressor(latent_x))
        coup_discrim = self.act(self.discriminant(discrim_x))
        
        x_cat = torch.cat([coup_encoder, coup_discrim], dim=-1)

        pi = self.act(self.pi(x_cat))
        mu = F.elu(self.mu(latent_x)) + 1.0
        sigma = self.sigma(latent_x)
        sigma = F.elu(sigma) + 1.1
        return pi, mu, sigma
    
    def gaussian_probability(self, sigma, mu, target, log=False):
        """Returns the probability of `target` given MoG parameters `sigma` and `mu`.
        Arguments:
            sigma (BxG): The standard deviation of the Gaussians. B is the batch
                size, G is the number of Gaussians.
            mu (BxG): The means of the Gaussians. B is the batch size, G is the
                number of Gaussians.
            target (BxI): A batch of target. B is the batch size and I is the number of
                input dimensions. Default I is 1.
        Returns:
            probabilities (BxG): The probability of each point in the probability
                of the distribution in the corresponding sigma/mu index.
        """
        if target.dim() == 1:
            target = target.unsqueeze(-1)
        target = target.expand_as(sigma)

        if log:
            return -torch.log(sigma) - 0.5 * LOG2PI - 0.5 * torch.pow((target - mu) / sigma, 2)

        else:
            return (ONEOVERSQRT2PI / sigma) * torch.exp(-0.5 * ((target - mu) / sigma) ** 2)

    def log_prob(self, pi, sigma, mu, y):
        log_component_prob = self.gaussian_probability(sigma, mu, y, log=True)
        log_mix_prob = torch.log(
            F.gumbel_softmax(pi, tau=1, dim=-1) + 1e-15
        )
        return torch.logsumexp(log_component_prob + log_mix_prob, dim=-1).unsqueeze(-1)

    def sample(self, pi, sigma, mu):
        """Draw samples from a MoG."""
        categorical = torch.distributions.Categorical(pi)
        pis = categorical.sample().unsqueeze(1)
        sample = torch.autograd.Variable(sigma.data.new(sigma.size(0), 1).normal_())
        # Gathering from the n Gaussian Distribution based on sampled indices
        sample = sample * sigma.gather(1, pis) + mu.gather(1, pis)
        return sample

    def generate_samples(self, pi, sigma, mu, n_samples=1):
        softmax_pi = F.gumbel_softmax(pi, tau=1, dim=-1)
        assert (
            softmax_pi < 0
        ).sum().item() == 0, "pi parameter should not be negative"
        samples = [self.sample(softmax_pi, sigma, mu) for _ in range(n_samples)]
        samples = torch.cat(samples, dim=1)
        return samples

    def generate_point_predictions(self, pi, sigma, mu, n_samples=1):
        # Sample using n_samples and take average
        samples = self.generate_samples(pi, sigma, mu, n_samples)
        return torch.mean(samples, dim=-1, keepdim=True)

    def calculate_loss(self, pi, mu, sigma, target):
        # nll loss for model training
        loss = -self.log_prob(pi, sigma, mu, target)
        return torch.mean(loss)

        

    
