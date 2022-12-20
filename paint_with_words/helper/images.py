def _img_importance_flatten(img: torch.tensor, ratio: int) -> torch.tensor:
    return F.interpolate(
        img.unsqueeze(0).unsqueeze(1),
        scale_factor=1 / ratio,
        mode="bilinear",
        align_corners=True,
    ).squeeze()
