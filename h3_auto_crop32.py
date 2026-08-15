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


class MiniMaxH3StartCanvasSelector:
    """Choose the H3 canvas from either the source-video crop or manual starter dimensions."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start_mode": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Connect H3 Extension Start Mode. load_video = use the cropped source-video canvas; generate_starter = use the manual width/height below."
                }),
                "generated_width": ("INT", {
                    "default": 960, "min": 32, "max": 16384, "step": 32,
                }),
                "generated_height": ("INT", {
                    "default": 544, "min": 32, "max": 16384, "step": 32,
                }),
            },
            "optional": {
                "source_width": ("INT", {
                    "lazy": True,
                    "tooltip": "Width from H3 Crop Source To /32. Requested only in load_video mode."
                }),
                "source_height": ("INT", {
                    "lazy": True,
                    "tooltip": "Height from H3 Crop Source To /32. Requested only in load_video mode."
                }),
            },
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "select"
    CATEGORY = "conditioning/minimax"
    DESCRIPTION = (
        "Choose one shared H3 generation canvas for the masked AV extension "
        "workflow. In load_video mode it reuses the cropped source-video size; "
        "in generate_starter mode it uses the manual starter width/height."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Input-signature caching already tracks start mode and dimensions.
        # Never poison every downstream sampler signature with NaN.
        return "h3-start-canvas-selector-v2"

    def check_lazy_status(self, start_mode, generated_width, generated_height, source_width=None, source_height=None):
        if str(start_mode) == "load_video":
            needed = []
            if source_width is None:
                needed.append("source_width")
            if source_height is None:
                needed.append("source_height")
            return needed
        return []

    def select(self, start_mode="load_video", generated_width=960, generated_height=544, source_width=None, source_height=None):
        if str(start_mode) == "load_video":
            if source_width is None or source_height is None:
                raise ValueError("H3 Start Canvas Selector: load_video mode needs source_width and source_height")
            return (int(source_width), int(source_height))
        return (int(generated_width), int(generated_height))
