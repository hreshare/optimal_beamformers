from re import U
import torch


'''
def mask_based_bf(x, mask, rescale_target):
    """
    x: observation spectrograms. shape: (bins, N or 1, mics, frames), where N is batch size.
    mask: time-frequency mask. shape: (bins, N, 1, frames)
    rescale_target: an observation used in rescaling. shape: (bins, N, 1, frames)

    output: BF result(s). shape: (bins, N, 1, frames)
    """
    frames = x.shape[-1]   # x.shape = (bins, channels, frames)
    cov_x = torch.matmul(x, htp(x)) / frames
    weighted_x = mask * x
    weighted_cov_x = torch.matmul(weighted_x, htp(x)) / frames
    w = gev_min(weighted_cov_x, cov_x)
    bf_out = torch.matmul(htp(w), x)
    bf_out = rescale(bf_out, rescale_target)
    return bf_out


def gev_min(wcov, cov):
    """
    Compute an eigenvector corresponding to the minimum eigenvalue
    """

    # Compute a decorrelation matrix P.
    D, Q = torch.linalg.eigh(cov)
    D = D.unsqueeze(-1)
    P = htp(Q) / D.sqrt()   # P = diag(D)^(-1/2) Q^H
    # In order to utilize eigh(), apply P to wcov.
    decorr_wcov = torch.matmul(torch.matmul(P, wcov), htp(P))
    _, vecs = torch.linalg.eigh(decorr_wcov)
    # return an eigenvector corresponding to the minimum eigenvalue.
    return vecs[...,:1]
'''

def mask_based_bf(x, mask, rescale_target):
    """
    x: observation spectrograms. shape: (bins, N or 1, mics, frames), where N is batch size.
    mask: time-frequency mask. shape: (bins, N, 1, frames)
    rescale_target: an observation used in rescaling. shape: (bins, N, 1, frames)

    output: BF result(s). shape: (bins, N, 1, frames)
    """
    frames = x.shape[-1]   # x.shape = (bins, channels, frames)
    u, _ = decorrelate(x)
    weighted_u = mask * u
    weighted_cov_u = torch.matmul(weighted_u, htp(u)) / frames
    _, vecs = torch.linalg.eigh(weighted_cov_u)
    w = vecs[...,:1]
    bf_out = torch.matmul(htp(w), u)
    bf_out = rescale(bf_out, rescale_target)
    return bf_out

def decorrelate(x):
    frames = x.shape[-1]   # x.shape = (bins, channels, frames)
    cov_x = torch.matmul(x, htp(x)) / frames
    D, Q = torch.linalg.eigh(cov_x)
    D = D.unsqueeze(-1)
    P = htp(Q) / D.sqrt()   # P = diag(D)^(-1/2) Q^H
    u = torch.matmul(P, x)
    return u, P

def htp(x):
    return torch.swapaxes(x, -1, -2).conj()


def rescale(y, rescale_target):
    cov = torch.matmul(rescale_target, htp(y))
    var = torch.matmul(y, htp(y))
    y_scaled = y * cov / var
    return y_scaled


