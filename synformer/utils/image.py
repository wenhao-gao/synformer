import math

from PIL import Image, ImageDraw, ImageFont


def make_grid(images: list[Image.Image]) -> Image.Image:
    """Make a grid of images.

    Args:
        images (list[PIL.Image.Image]): A list of images.

    Returns:
        PIL.Image.Image: A grid of images.
    """
    width = max(image.size[0] for image in images)
    height = max(image.size[1] for image in images)

    num_cols = int(math.ceil(math.sqrt(len(images))))
    num_rows = int(math.ceil(len(images) / num_cols))
    grid = Image.new("RGB", (num_cols * width, num_rows * height), color=(255, 255, 255))
    for i, image in enumerate(images):
        x = width * (i % num_cols) + (width - image.size[0]) // 2
        y = height * (i // num_cols) + (height - image.size[1]) // 2
        grid.paste(image, (x, y))
    return grid


def draw_text(
    message: str,
    W: int = 100,
    H: int = 100,
    size: int = 20,
    bg_color: tuple[int, int, int] = (73, 109, 137),
) -> Image.Image:
    image = Image.new("RGB", (W, H), color=bg_color)
    draw = ImageDraw.Draw(image)
    # font = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", size)
    font = ImageFont.load_default()
    _, _, w, h = draw.textbbox((0, 0), message, font=font)
    draw.text(((W - w) / 2, (H - h) / 2), message, font=font)
    return image
