import torch


class MiniMaxH3CropTo32:
    """Center-crop an IMAGE batch down to the nearest dimensions divisible by 32."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Source video frames. Width and height are cropped DOWN to the nearest multiple of 32."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("images", "width", "height")
    FUNCTION = "crop_to_32"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = (
        "Center-crop source video frames down to the nearest width and height "
        "divisible by 32 and output the resulting dimensions for the H3 target."
    )

    def crop_to_32(self, images):
        if getattr(images, "ndim", 0) != 4:
            raise ValueError("H3 Crop Source To /32 expects IMAGE [N,H,W,C].")

        h = int(images.shape[1])
        w = int(images.shape[2])

        target_w = (w // 32) * 32
        target_h = (h // 32) * 32

        if target_w < 32 or target_h < 32:
            raise ValueError(
                f"Source is too small ({w}x{h}); cropped H3 dimensions must be at least 32x32."
            )

        crop_x = w - target_w
        crop_y = h - target_h
        left = crop_x // 2
        top = crop_y // 2

        cropped = images[:, top:top + target_h, left:left + target_w, :]

        print(
            f"[H3 Motion Context] Source crop /32: "
            f"{w}x{h} -> {target_w}x{target_h} "
            f"(left={left}, top={top})"
        )

        return (cropped, target_w, target_h)
