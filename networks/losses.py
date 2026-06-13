import torch
import torch.nn as nn

class ContentLoss(nn.Module):
    def __init__(self, vgg_model, weight_sparse=1e-4):
        super().__init__()
        self.vgg = vgg_model
        self.l1 = nn.L1Loss()
        self.weight_sparse = weight_sparse

    def forward(self, generated, target):
        feat_gen = self.vgg(generated)
        feat_tar = self.vgg(target)
        
        # Standard Perceptual Loss
        loss_content = self.l1(feat_gen, feat_tar.detach())
        
        # Sparse Regularization: Prevents color bleeding and preserves semantics
        loss_sparse = torch.mean(torch.abs(feat_gen))
        
        return loss_content + self.weight_sparse * loss_sparse

class EdgePromotingAdversarialLoss(nn.Module):
    """
    The discriminator learns to distinguish real sharp cartoons (label 1)
    from smoothed cartoons (label 0). This forces the generator to produce
    sharp edges to fool the discriminator.
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward_D(self, d_real, d_smoothed, d_generated):
        loss_real = self.bce(d_real, torch.ones_like(d_real))
        loss_smoothed = self.bce(d_smoothed, torch.zeros_like(d_smoothed))
        loss_gen = self.bce(d_generated, torch.zeros_like(d_generated))
        return loss_real + loss_smoothed + loss_gen

    def forward_G(self, d_generated):
        return self.bce(d_generated, torch.ones_like(d_generated))