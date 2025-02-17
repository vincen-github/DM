from torch import randn, acos, tensor, randperm, randint, cat, zeros, arange, int8, float32
from torch.nn.functional import normalize
from torch.cuda import is_available
from numpy import full
from numpy.random import choice

device = "cuda" if is_available() else "cpu"

def gen_centers(num_centers, dim):
    centers = zeros((num_centers, dim), dtype=float32).to(device)
    indices = randperm(dim)[:num_centers]
    centers[arange(num_centers), indices] = randint(0, 2, (num_centers,), dtype=float32).to(device) * 2 - 1
    return centers

def perturbation(template, n, eps):
    dim = template.size(0)
    
    perturbation_dir = normalize(randn((n, dim)), dim=1).to(device)
    
    return normalize(template + eps * perturbation_dir)

def gen_reference(centers, n, eps):
    num_centers = centers.size(0)
    
    base = n // num_centers 
    remainder = n % num_centers
    
    parts = full(shape=num_centers, fill_value=base)
    
    for i in choice(a=num_centers, size=remainder, replace=False):
        parts[i] += 1
    
    reference = []
    for i in range(num_centers):
        reference.append(perturbation(centers[i], parts[i], eps))
    return cat(reference, dim=0)
