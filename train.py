from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingWarmRestarts
import torch.backends.cudnn as cudnn
from methods.utils import gen_centers
from model import Critic
from numpy import mean
from cfg import get_cfg
from datasets import get_ds
from methods import get_method
from torch import save, load

if __name__ == "__main__":
    cfg = get_cfg()

    ds = get_ds(cfg.dataset)(cfg.bs, cfg, cfg.num_workers)

    centers = gen_centers(cfg.Kprime, cfg.emb)
    model = get_method(cfg.method)(cfg, centers)

    critic = Critic(cfg.emb)
    model.cuda().train()
    critic.cuda().train()
    if cfg.fname is not None:
        model.load_state_dict(load(cfg.fname))

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.99), weight_decay=cfg.adam_l2)
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3, betas=(0.9, 0.99), weight_decay=1e-4)

    eval_every = cfg.eval_every
    lr_warmup = 0 if cfg.lr_warmup else 500
    cudnn.benchmark = True

    eps = cfg.eps

    for ep in range(cfg.epoch):
        loss_ep = []
        critic_loss_ep = []
        iters = len(ds.train)
        for n_iter, (samples, _) in enumerate(tqdm(ds.train, position=1)):
            if lr_warmup < 500:
                lr_scale = (lr_warmup + 1) / 500
                for pg in optimizer.param_groups:
                    pg["lr"] = cfg.lr * lr_scale
                lr_warmup += 1

            for i in range(5):
                critic_optimizer.zero_grad()
                _, critic_loss = model(samples, critic, eps)
                critic_loss.backward(retain_graph=True)
                critic_optimizer.step()
                if i == 4:
                    critic_loss_ep.append(critic_loss.item())
                
            optimizer.zero_grad()
            loss, _ = model(samples, critic, eps)
            loss.backward()
            optimizer.step()
            loss_ep.append(loss.item())
            
        if (ep + 1) % eval_every == 0:
            acc_knn, acc = model.get_acc(ds.clf, ds.test)
            print({"acc": acc[1], "acc_5": acc[5], "acc_knn": acc_knn})

        if (ep + 1) % 100 == 0:
            fname = f"models/{cfg.method}_{cfg.dataset}_{ep}_{acc[1]}_{acc_knn}.pt"
            save(model.state_dict(), fname)

        print({"loss": mean(loss_ep), "critic_loss": mean(critic_loss_ep), "ep": ep})

