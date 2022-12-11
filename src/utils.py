import numpy as np
import random
import colorsys

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import Polygon
import matplotlib.patheffects as PathEffects
from skimage.measure import find_contours

from PIL import Image

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """
    Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1, image[:, :, c] * (1 - alpha) + alpha * color[c] * 255, image[:, :, c])
    return image


def display_instances(image, boxes, masks, class_ids, class_names, figsize=(16, 16),
                      scores=None, title="", ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None, confidence=0.7):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)

    # Generate random colors
    colors = colors or random_colors(N)

    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        if scores is not None and scores[i] > confidence:
            color = colors[i]

            # Bounding box
            if not np.any(boxes[i]):
                # Skip this instance. Has no bbox. Likely lost in image cropping.
                continue
            y1, x1, y2, x2 = boxes[i]
            if show_bbox:
                p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=3,
                                      alpha=0.8, linestyle='dashed',
                                      edgecolor=color, facecolor='none')
                ax.add_patch(p)

            # Label
            if not captions:
                class_id = class_ids[i]
                score = None
                label = class_names[class_id]
                caption = "{} {:.3f}".format(label, score) if score else label
            else:
                caption = captions[i]
            h, l, s = colorsys.rgb_to_hls(*color)

            # manipulate h, l, s values and return as rgb
            text_color = colorsys.hls_to_rgb(h, min(1, l * 1.1), s=s)
            t = ax.text(0.5 * (x1 + x2), 0.5 * (y1 + y2), caption, horizontalalignment='center',
                        verticalalignment='center', color=text_color, size=28)
            t.set_path_effects([PathEffects.withStroke(linewidth=6, foreground='black')])

            # Mask
            mask = masks[:, :, i]
            if show_mask:
                masked_image = apply_mask(masked_image, mask, color)

            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8
            )
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor='black')
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    plt.savefig("result.png")
    crop("result.png")


def crop(filename):
    file = Image.open(filename)
    # Setting the points for cropped image
    left = 460
    top = 200
    right = 1210
    bottom = 1420

    # Cropped image of above dimension
    # (It will not change original image)
    im1 = file.crop((left, top, right, bottom))

    # Save the image
    im1.save(filename)

def rotate_bbox(bbox, angle, size):
    """
    Rotate bounding boxes by degrees.
    Args:
        bbox (~numpy.ndarray): An array whose shape is :math:`(R, 4)`.
            :math:`R` is the number of bounding boxes.
        angle (float): Counter clock-wise rotation angle (degree).
            image is rotated by 90 degrees.
        size (tuple): A tuple of length 2. The height and the width
            of the image.

    Returns:
        ~numpy.ndarray:
        Bounding boxes rescaled according to the given :obj:`k`.
    """
    if angle % 90 != 0:
        raise ValueError(
            'angle which satisfies angle % 90 == 0 is only supported: {}'
            .format(angle)
        )
    H, W = size
    if angle % 360 == 0:
        return bbox

    if angle % 360 == 90:
        rotated_bbox = np.concatenate(
            (W - bbox[:, 3:4], bbox[:, 0:1],
             W - bbox[:, 1:2], bbox[:, 2:3]), axis=1)
    elif angle % 360 == 180:
        rotated_bbox = np.concatenate(
            (H - bbox[:, 2:3], W - bbox[:, 3:4],
             H - bbox[:, 0:1], W - bbox[:, 1:2]), axis=1)
    elif angle % 360 == 270:
        rotated_bbox = np.concatenate(
            (bbox[:, 1:2], H - bbox[:, 2:3],
             bbox[:, 3:4], H - bbox[:, 0:1]), axis=1)
    rotated_bbox = rotated_bbox.astype(bbox.dtype)
    return rotated_bbox

