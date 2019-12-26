import numpy as np
from PIL import Image, ImageDraw

ALPHA = 128
COLORS = dict(R=(255,0,0), G=(0,255,0), B=(0,0,255), P=(255,0,255))
BG, FG = None, None

HOLD_TOPLEFT = (93, 87)  # x, y in pixels; not row, col
HOLD_SPACING = 50  # px
MAX_RADIUS = 60
MIN_RADIUS = 5

def usage_to_image(usage, color, show=True):
    assert (usage.shape == (18, 11)), "Usage array shape must be (18, 11)"
    # Get min/max for pretty radius calculations
    min_val, max_val = np.ma.masked_equal(usage, 0.0, copy=False).min(), usage.max()  # Nonzero minimum, maximum (https://stackoverflow.com/questions/7164397/find-the-min-max-excluding-zeros-in-a-numpy-array-or-a-tuple-in-python)
    slope = (MAX_RADIUS - MIN_RADIUS) / (max_val - min_val)
    # Variables for PIL
    color = COLORS[color]
    fg, bg = _imgs()
    draw = ImageDraw.Draw(fg)
    # Draw circles
    for r in range(0, 18):
        for c in range(0, 11):
            val = usage[r, c]
            if val:
                rad = slope * (val - min_val) + MIN_RADIUS
                coords = _rc_to_xy(r, c)
                _circle(draw, coords, rad, color, fill=True, width=2)
    # Return
    img = Image.alpha_composite(bg, fg)
    if show:
        if get_ipython():
            display(img)
        else:
            img.show()
    return img

def array_to_image(p, rad=30, show=False):
    fg, bg = _imgs()
    draw = ImageDraw.Draw(fg)
    for r in range(0, 18):
        for c in range(0, 11):
            if not p[r, c, 3]:  # Not unused
                if p[r, c, 0]:  # End
                    color = 'R'
                elif p[r, c, 1]:  # Start
                    color = 'G'
                elif p[r, c, 2]:  # Others
                    color = 'B'
                _circle(draw, _rc_to_xy(r, c), rad, COLORS[color], fill=False)
    img = Image.alpha_composite(bg, fg)
    if show:
        if get_ipython():
            display(img)
        else:
            img.show()
    return img

def _imgs(filename='mb2016.png'):
    global FG, BG
    if not FG or BG:
        BG = Image.open(filename).convert('RGBA')
        FG = Image.new('RGBA', BG.size, (255, 255, 255, 0))
    return FG.copy(), BG

def _circle(draw, center, rad, color, fill=False, width=5):
    bounds = [center[0] - rad, center[1] - rad, center[0] + rad, center[1] + rad]
    if fill:
        draw.ellipse(bounds, fill=(color[0], color[1], color[2], ALPHA))
    draw.ellipse(bounds, outline=(color[0], color[1], color[2], 255), width=width)

def _rc_to_xy(r, c):
    origin_x, origin_y = HOLD_TOPLEFT
    return origin_x + c * HOLD_SPACING, origin_y + r * HOLD_SPACING