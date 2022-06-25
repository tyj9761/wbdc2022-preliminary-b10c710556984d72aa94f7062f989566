import logging
import os
import time
import torch

from tqdm import tqdm
from config import parse_args
from data_helper import create_dataloaders
from model import MultiModal
from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate
from torch import nn

def validate(model, val_dataloader):
    model.eval()
    predictions = []
    labels = []
    losses = []
    with torch.no_grad():
        loop2 = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
        for _,batch in loop2:
            loss, _, pred_label_id, label = model(batch)
            loss = loss.mean()
            predictions.extend(pred_label_id.cpu().numpy())
            labels.extend(label.cpu().numpy())
            losses.append(loss.cpu().numpy())
    loss = sum(losses) / len(losses)
    results = evaluate(predictions, labels)

    model.train()
    return loss, results


def train_and_validate(args,fold):
    # 1. load data
    train_dataloader, val_dataloader = create_dataloaders(args,fold)

    # 2. build model and optimizers
    model = MultiModal(args)

    ema = EMA(model, 0.999)
    ema.register()
    optimizer, scheduler = build_optimizer(args, model)
    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))

    fgm = FGM(model)

    # 3. training
    step = 0
    best_score = args.best_score
    start_time = time.time()
    num_total_steps = len(train_dataloader) * args.max_epochs
    for epoch in range(args.max_epochs):
        loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for _,batch in loop:
            model.train()
            loss, accuracy, _, _ = model(batch)
            loss = loss.mean()
            accuracy = accuracy.mean()
            loss.backward()

            fgm.attack()
            optimizer.zero_grad()
            loss_adv = model(batch)[0]
            loss_adv.backward()
            fgm.restore()

            optimizer.step()
            ema.update()
            optimizer.zero_grad()
            scheduler.step()

            step += 1
            spend_time = time.time() - start_time
            time_per_step =  spend_time/ max(1, step)
            remaining_time = time_per_step * (num_total_steps - step)
            remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
            loop.set_description(f'Epoch [{epoch+1}/{args.max_epochs}]')
            loop.set_postfix(acc=float(accuracy),eta=remaining_time,loss=float(loss),)
            loop.update()

                # logging.info(f"Epoch {epoch} step {step}  {remaining_time} eta {remaining_time}: loss {loss:.3f}, accuracy {accuracy:.3f}")

        # 4. validation
        ema.apply_shadow()
        loss, results = validate(model, val_dataloader)
        results = {k: round(v, 4) for k, v in results.items()}

        logging.info(f"Epoch {epoch+1}  loss {loss:.3f}, {results}")

        # 5. save checkpoint
        mean_f1 = results['mean_f1']
        if mean_f1 > best_score:
            best_score = mean_f1
            torch.save({'epoch': epoch, 'model_state_dict': model.module.state_dict(), 'mean_f1': mean_f1},
                       f'{args.savedmodel_path}/model_epoch_{epoch}.bin')
        ema.restore()


class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name].to('cuda')
                new_average = new_average.to('cuda')
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class FGM:
    def __init__(self, model: nn.Module, eps=1.):
        self.model = (
            model.module if hasattr(model, "module") else model
        )
        self.eps = eps
        self.backup = {}

    # only attack word embedding
    def attack(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm and not torch.isnan(norm):
                    r_at = self.eps * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        for name, para in self.model.named_parameters():
            if para.requires_grad and emb_name in name:
                assert name in self.backup
                para.data = self.backup[name]

        self.backup = {}

def main():
    args = parse_args()
    setup_logging()
    setup_device(args)
    setup_seed(args,0)

    os.makedirs(args.savedmodel_path, exist_ok=True)
    logging.info("Training/evaluation parameters: %s", args)

    train_and_validate(args,fold=0)


if __name__ == '__main__':
    main()
