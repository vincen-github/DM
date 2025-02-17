from torch.nn.functional import normalize
from .base import BaseMethod
from torch import cat, norm, rand, ones, mean
from torch.autograd import grad
from .utils import gen_reference

def dm(x0, x1, critic, centers, eps, lambda_param=1):
    # normalize x0 and x1
    x0 = normalize(x0)
    x1 = normalize(x1)
    
    # Get the number of augmentations and the dimension
    N = x0.size(0)
    D = x0.size(1)
    
    x = cat([x0, x1], dim=0)
    
    reference = gen_reference(centers, 2 * N, eps)
    
    alpha = rand(1).repeat(2 * N, D).to(x0.device)
    interpolates = alpha * x + (1 - alpha) * reference
    
    critic_score = critic(interpolates).reshape(-1)
    gradients = grad(
                outputs=critic_score,
                inputs=interpolates,
                grad_outputs=ones(2 * N).to(x0.device),
                create_graph=True,
                retain_graph=True
            )[0]
    gp = mean(pow(gradients.norm(2, dim=1) - 1, 2))
    
    wasserstein = mean(critic(reference)) - mean(critic(x))
    
    return norm(x0 - x1, p=2, dim=1).pow(2).mean() + lambda_param * wasserstein, -wasserstein + gp

class DM(BaseMethod):
    """ implements our ssl loss"""
    def __init__(self, cfg, centers):
        super().__init__(cfg)
        self.loss_f = dm
        self.centers = centers

    def forward(self, samples, critic, eps):
        bs = len(samples[0])
        h = [self.model(x.cuda(non_blocking=True)) for x in samples]
        h = self.head(cat(h))
        loss = 0
        critic_loss = 0
        for i in range(len(samples) - 1):
            for j in range(i + 1, len(samples)):
                x0 = h[i * bs: (i + 1) * bs]
                x1 = h[j * bs: (j + 1) * bs]
                returns = self.loss_f(x0, x1, critic, self.centers, eps)
                loss += returns[0]
                critic_loss += returns[1]
        loss /= self.num_pairs
        critic_loss /= self.num_pairs
        return loss, critic_loss
