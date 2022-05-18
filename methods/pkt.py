import torch
import numpy as np


def pairwise_distances(a, b=None, eps=1e-6):
    """
    Calculates the pairwise distances between matrices a and b (or a and a, if b is not set)
    :param a:
    :param b:
    :return:
    """
    if b is None:
        b = a

    aa = torch.sum(a ** 2, dim=1)
    bb = torch.sum(b ** 2, dim=1)

    aa = aa.expand(bb.size(0), aa.size(0)).t()
    bb = bb.expand(aa.size(0), bb.size(0))

    AB = torch.mm(a, b.transpose(0, 1))

    dists = aa + bb - 2 * AB
    dists = torch.clamp(dists, min=0, max=np.inf)
    dists = torch.sqrt(dists + eps)
    return dists


def cosine_pairwise_similarities(features, eps=1e-6, normalized=True):
    features_norm = torch.sqrt(torch.sum(features ** 2, dim=1, keepdim=True))
    features = features / (features_norm + eps)
    features[features != features] = 0
    similarities = torch.mm(features, features.transpose(0, 1))

    if normalized:
        similarities = (similarities + 1.0) / 2.0
    return similarities


def PKT_loss(teacher_features, student_features):
    # Teacher kernel
    
    teacher_d = pairwise_distances(teacher_features)
    
    d = 1
    teacher_s_2 = 1.0 / (1 + teacher_d ** d)
    teacher_s_1 = cosine_pairwise_similarities(teacher_features)


    # Student kernel
    
    student_d = pairwise_distances(student_features)
    d = 1
    student_s_2 = 1.0 / (1 + student_d ** d)
    student_s_1 = cosine_pairwise_similarities(student_features)



    # Transform them into probabilities
    teacher_s_1 = teacher_s_1 / torch.sum(teacher_s_1, dim=1, keepdim=True)
    student_s_1 = student_s_1 / torch.sum(student_s_1, dim=1, keepdim=True)
    teacher_s_2 = teacher_s_2 / torch.sum(teacher_s_2, dim=1, keepdim=True)
    student_s_2 = student_s_2 / torch.sum(student_s_2, dim=1, keepdim=True)

    

    # Jeffrey's  combined
    loss1 = (teacher_s_1 - student_s_1) * (torch.log(teacher_s_1) - torch.log(student_s_1))
    
    loss2 = (teacher_s_2 - student_s_2) * (torch.log(teacher_s_2) - torch.log(student_s_2))


    loss = torch.mean(loss1) + torch.mean(loss2)

    return loss
