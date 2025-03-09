# -*- coding: utf-8 -*-

"""
DeepInversion

credits:
    https://github.com/NVlabs/DeepInversion
    https://github.com/GT-RIPL/AlwaysBeDreaming-DFCIL
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange

from ..nn.module import freeze, unfreeze

from .feature_hook import DeepInversionFeatureHook
from .gaussian_smoothing import Gaussiansmoothing
from .generator import create as gen_create


def cov(X):
    D = X.shape[-1]
    mean = torch.mean(X, dim=-1).unsqueeze(-1)
    X = X - mean
    return 1/(D-1) * X @ X.transpose(-1, -2)

def get_means_cov(features):
    sample_class_mean = [i.mean(dim=0, keepdim=True) for i in features]
    dif = torch.cat([i-j for (i,j) in zip(features, sample_class_mean) if not torch.isnan(j.mean())], dim=0)
    assert not torch.any(torch.isnan(dif))
    covariance = cov(dif.T)
    return torch.cat(sample_class_mean, dim=0), covariance


def split_class_features(features, targets, num_classes):
    class_features = []
    classes = []
    for i in range(num_classes):
        index = targets.eq(i)
        if torch.any(index):
            classes.append(True)
            class_features.append(features[index])
        else:
            classes.append(False)
    return class_features, torch.tensor(classes, device=features.device)


class ClassMeanVar(nn.Module):
    def __init__(self, datamodule, model: nn.Module, generator):
        super().__init__()
        self.datamodule = datamodule
        self.num_classes = model.head.num_classes
        self.num_old_classes = self.num_classes - datamodule.num_classes
        self.model = model
        self.generator = generator

        self.class_mean, self.covariance = self.estimate_teacher()

    def estimate_teacher(self):
        features = [[] for _ in range(self.num_classes)]

        dataloader = self.datamodule.train_dataloader()
        self.model.eval()
        for data, target in dataloader:
            data, target = data.cuda(), target.cuda()
            target = self.datamodule.transform_target(target)
            with torch.no_grad():
                self.model.head.feature_mode = True
                out_features = self.model(data)
                self.model.head.feature_mode = False
            for i in range(self.num_classes):
                features[i].append(out_features[target.eq(i)])

        if self.generator is not None:
            batch_size = self.datamodule.batch_size
            self.generator.eval()
            for _ in range(len(dataloader) * int(self.num_old_classes / self.datamodule.num_classes)):
                input_rh, target_rh = self.generator.sample(batch_size)
                with torch.no_grad():
                    self.model.head.feature_mode = True
                    out_features = self.model(input_rh)
                    self.model.head.feature_mode = False
                for i in range(self.num_classes):
                    features[i].append(out_features[target_rh.eq(i)])

        for (i, feature) in enumerate(features):
            features[i] = torch.cat(feature, dim=0)

        return get_means_cov(features)


class GenerativeInversion(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        dataset: str,
        input_dims: Tuple[int, int, int] = (3, 32, 32),
        batch_size: int = 256,
        max_iters: int = 5000,
        lr: float = 1e-3,
        tau: float = 1e3,
        alpha_ce: float = 1.,
        alpha_cb: float = 1.,
        alpha_pr: float = 1e-3,
        alpha_rf: float = 5.0,
        alpha_gem: float = 1.,
        bn_mmt = 0.9,
        meanvar = None,
        writer = None
    ):
        super().__init__()

        self.batch_size = batch_size
        self.max_iters = max_iters
        self.lr = lr
        self.tau = tau
        self.alpha_ce = alpha_ce
        self.alpha_cb = alpha_cb
        self.alpha_pr = alpha_pr
        self.alpha_rf = alpha_rf
        self.alpha_gem = alpha_gem
        self.feature_hooks = []

        self.model = model
        self.generator = gen_create(dataset)
        self.smoothing = Gaussiansmoothing(3, 5, 1)
        self.criterion_ce = nn.CrossEntropyLoss()

        self.teacher_class_mean, self.teacher_covariance = meanvar.class_mean, meanvar.covariance
        self.running_class_mean, self.running_covariance = torch.zeros_like(self.teacher_class_mean), torch.zeros_like(self.teacher_covariance)
        self.bn_mmt = bn_mmt
        self.index = torch.tensor([False]*meanvar.num_classes, device=self.teacher_class_mean.device)

        self.writer = writer

    def setup(self):
        freeze(self.model)
        self.register_feature_hooks()

    def register_feature_hooks(self):
        # Remove old before register
        for hook in self.feature_hooks:
            hook.remove()

        ## Create hooks for feature statistics catching
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.feature_hooks.append(DeepInversionFeatureHook(module))

    def criterion_pr(self, input):
        input_pad = F.pad(input, (2, 2, 2, 2), mode="reflect")
        input_smooth = self.smoothing(input_pad)
        return F.mse_loss(input, input_smooth)

    def criterion_rf(self):
        #  return sum([hook.r_feature for hook in self.feature_hooks])
        return torch.stack([h.r_feature for h in self.feature_hooks]).mean()

    def criterion_cb(self, output: torch.Tensor):
        logit_mu = output.softmax(dim=1).mean(dim=0)
        num_classes = output.shape[1]
        # ignore sign
        entropy = (logit_mu * (logit_mu+1e-6).log() / math.log(num_classes)).sum()
        return 1 + entropy

    @torch.no_grad()
    def sample(self, batch_size: int = None):
        _ = self.model.eval() if self.model.training else None
        batch_size = self.batch_size if batch_size is None else batch_size
        input = self.generator.sample(batch_size)
        target = self.model(input).argmax(dim=1)
        return input, target

    def train_step(self):
        input = self.generator.sample(self.batch_size)
        self.model.head.feature_mode = True
        out_features = self.model(input)
        self.model.head.feature_mode = False
        output = self.model.head.classify(out_features)
        target = output.data.argmax(dim=1)

        # content loss
        loss_ce = self.criterion_ce(output / self.tau, target)

        # label diversity loss
        loss_cb = self.criterion_cb(output)

        # locally smooth prior
        loss_pr = self.criterion_pr(input)

        # feature statistics regularization
        loss_rf = self.criterion_rf()

        # class statistics regularization
        assert self.running_class_mean.requires_grad is False
        class_features, classes = split_class_features(out_features, target, self.model.head.num_classes)
        class_mean, covariance = get_means_cov(class_features)
        if not torch.all(self.index):
            running_class_mean, running_covariance = class_mean, covariance
        else:
            running_class_mean = self.running_class_mean[classes] * self.bn_mmt + class_mean * (1 - self.bn_mmt)
            running_covariance = self.running_covariance * self.bn_mmt + covariance * (1 - self.bn_mmt)

        mask = torch.logical_not(torch.isnan(self.teacher_class_mean[classes].mean(dim=1)))
        assert mask.requires_grad is False
        assert not torch.any(torch.isnan(self.teacher_class_mean[classes][mask]))
        loss_gem = torch.norm((self.teacher_class_mean[classes] - running_class_mean)[mask]) + \
                   torch.norm(self.teacher_covariance - running_covariance)

        loss = self.alpha_ce * loss_ce + self.alpha_cb * loss_cb + self.alpha_pr * loss_pr + self.alpha_rf * loss_rf + \
               self.alpha_gem * loss_gem

        loss_dict = {
            "ce": loss_ce,
            "cb": loss_cb,
            "pr": loss_pr,
            "rf": loss_rf,
            "gem": loss_gem,
            "total": loss,
        }

        if self.bn_mmt != 0:
            self.running_class_mean[classes], self.running_covariance = running_class_mean.data, running_covariance.data
            self.index = torch.logical_or(self.index, classes)

        return loss, loss_dict

    def configure_optimizers(self):
        params = self.generator.parameters()
        return optim.Adam(params, lr=self.lr)

    def forward(self):
        _ = self.setup(), unfreeze(self.generator)
        optimizer = self.configure_optimizers()
        miniters = max(self.max_iters // 25, 1)
        pbar = trange(self.max_iters, miniters=miniters, desc="Inversion")
        for current_iter in pbar:
            loss, loss_dict = self.train_step()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (current_iter + 1) % miniters == 0:
                pbar.set_postfix({k: f"{v:.4f}" for k, v in loss_dict.items()})
                for i, (k, v) in enumerate(loss_dict.items()):
                    identity = str(i+1) if k != 'total' else '0'
                    self.writer.add_scalar('generator/'+identity+'.'+'loss_'+k, v.item(), current_iter)
        freeze(self.generator)
