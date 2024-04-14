import torch
import torch.nn.functional as F

class ArcFaceLoss(torch.nn.Module):
    def __init__(self, s=30.0, m=0.50, easy_margin=False):
        super(ArcFaceLoss, self).__init__()
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        """
        Args:
            logits: output logits, a tensor of dimension (batch_size, num_classes)
            labels: ground-truth labels, a tensor of dimension (batch_size)
        """
        # Normalize the input logits
        logits = F.normalize(logits, p=2, dim=1)

        # Compute the cosine similarity between each input and all the class centers
        cos_theta = logits.mm(logits.transpose(1, 0))

        # Get the cosine similarity of the true class
        cos_theta_y = cos_theta[torch.arange(0, logits.size(0)), labels].unsqueeze(1)

        # Compute the margin-modified cosine similarity
        cos_theta_m = cos_theta_y - self.m

        if self.easy_margin:
            cos_theta_m = torch.clamp(cos_theta_m, min=0)
        else:
            cos_theta_m = torch.clamp(cos_theta_m, -1 + self.m, 1 - self.m)

        # Compute the final logits
        output = cos_theta.clone()
        output[torch.arange(0, logits.size(0)), labels] = cos_theta_m

        # Compute the loss
        loss = self.cross_entropy(self.s * output, labels)
        return loss