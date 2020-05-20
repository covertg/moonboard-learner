import numpy as np
from PIL import Image, ImageDraw

FILL_ALPHA = 128
COLORS = dict(R=(255,0,0), G=(0,255,0), B=(0,0,255), P=(127,0,127))
_BG, _FG = None, None

HOLD_TOPLEFT = (93, 87)  # x, y in pixels; not row, col
HOLD_SPACING = 50  # px
MAX_RADIUS = 60
MIN_RADIUS = 5


def draw_problem(problem, rad=30, show=True):
    fg, bg = _base_imgs()
    draw = ImageDraw.Draw(fg)
    arr = problem.array_3d
    for (row, col, depth), hold in np.ndenumerate(arr):
        if hold:
            color = ['G', 'B', 'R'][depth]  # 0=start, 1=intermed, 2=end
            _circle(draw, _rc_to_xy(row, col), rad, COLORS[color], fill=False)                
    img = Image.alpha_composite(bg, fg)
    # Show image if show=True, return image if show=False
    if show:
        if get_ipython():
            display(img)
        else:
            img.show()
    else:
        return img


def draw_usage(problems, show=True):
    usages = _calc_usage(problems)
    colors = ['G', 'B', 'R', 'P']  # calc_usage returns start, intermed, end, combined
    imgs = []
    for usage_arr, color in zip(usages, colors):
        # PIL variables
        fg, bg = _base_imgs()
        draw = ImageDraw.Draw(fg)
        # min/max for pretty radius calculations
        min_usage, max_usage = np.ma.masked_equal(usage_arr, 0.0, copy=False).min(), usage_arr.max()  # Nonzero minimum, maximum (https://stackoverflow.com/questions/7164397/find-the-min-max-excluding-zeros-in-a-numpy-array-or-a-tuple-in-python)
        slope = (MAX_RADIUS - MIN_RADIUS) / (max_usage - min_usage)
        # Draw filled circles
        for (row, col), usage in np.ndenumerate(usage_arr):
            if usage:
                rad = slope * (usage - min_usage) + MIN_RADIUS
                _circle(draw, _rc_to_xy(row, col), rad, COLORS[color], fill=True, width=2)
        img = Image.alpha_composite(bg, fg)
        imgs.append(img)
    if show:
        if get_ipython():
            for img in imgs: display(img)
        else:
            for img in imgs: img.show()
    else:
        return img


# "Usage" is the proportion of times a given hold is used in a given set.
# Calculates for start holds, intermed holds, end holds, and all (flattened) holds.
def _calc_usage(problems):
    prob_arrs_3d = np.array([p.array_3d for p in problems])
    counts = np.sum(prob_arrs_3d, axis=0)  # (18, 11, 3)
    counts_combined = np.sum(counts, axis=-1)
    usages = [counts[:, :, 0] / np.sum(counts[:, :, 0]),
              counts[:, :, 1] / np.sum(counts[:, :, 1]),
              counts[:, :, 2] / np.sum(counts[:, :, 2]),
              counts_combined / np.sum(counts_combined)]
    return usages


def _base_imgs(filename='mb2016.png'):
    global _FG, _BG  # Without global keyword, we would be referencing without assignment
    if not (_FG or _BG):
        _BG = Image.open(filename).convert('RGBA')
        _FG = Image.new('RGBA', _BG.size, (255, 255, 255, 0))
    return _FG.copy(), _BG


def _circle(draw, center, rad, color, fill=False, width=5):
    bounds = [center[0] - rad, center[1] - rad, center[0] + rad, center[1] + rad]
    if fill:
        draw.ellipse(bounds, fill=(color[0], color[1], color[2], FILL_ALPHA))
    draw.ellipse(bounds, outline=(color[0], color[1], color[2], 255), width=width)


def _rc_to_xy(r, c):
    origin_x, origin_y = HOLD_TOPLEFT
    return origin_x + c * HOLD_SPACING, origin_y + r * HOLD_SPACING