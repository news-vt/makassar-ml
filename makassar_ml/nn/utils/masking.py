import torch


# Note that for Transformer decoder target mask,
# the `length` parameter is the desired length of
# the target sequence.
# See docs: https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html#torch.nn.Transformer.forward
def create_attn_mask(length: int):
    """Generate mask used for attention mechanisms.

    Masks are a lower-triangular matrix of zeros
    with the other entries taking value "-inf".

    Args:
        length (int): Length of square-matrix dimension.

    Examples:
        >>> create_attn_mask(3)
        tensor([[0., -inf, -inf],
                [0., 0., -inf],
                [0., 0., 0.]])
    """
    # Get lower-triangular matrix of ones.
    mask = torch.tril(torch.ones(length, length))

    # Replace 0 -> "-inf" and 1 -> 0.0
    mask = (
        mask
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask