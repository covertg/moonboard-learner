from PIL import Image, ImageDraw

BG, FG = None, None
HOLD_TOPLEFT = (93, 87)  # x, y in pixels; not row, col
HOLD_SPACING = 50  # px
A = 128
COLORS = dict(R=(255,0,0,A), G=(0,255,0,A), B=(0,0,255,A), W=(255,255,255,A))

def _imgs(filename='mb2016.png'):
    global FG, BG
    if not FG or BG:
        BG = Image.open(filename).convert('RGBA')
        FG = Image.new('RGBA', BG.size, (255, 255, 255, 0))
    return FG.copy(), BG

def _circle(draw, center, rad, color, fill=False):
    bounds = [center[0] - rad, center[1] - rad, center[0] + rad, center[1] + rad]
    if fill:
        draw.ellipse(bounds, fill=color)
    else:
        draw.ellipse(bounds, outline=color, width=3)

def _rc_to_xy(r, c):
    origin_x, origin_y = HOLD_TOPLEFT
    return origin_x + c * HOLD_SPACING, origin_y + r * HOLD_SPACING

def usage_to_image(usage, color, show=True):
    assert (usage.shape == (18, 11)), "Usage array shape must be (18, 11)"
    color = COLORS[color]
    fg, bg = _imgs()
    draw = ImageDraw.Draw(fg)
    for r in range(0, 18):
        for c in range(0, 11):
            coord = _rc_to_xy(r, c)
            if usage[r, c]:
                _circle(draw, coord, 10, color, fill=True)
    img = Image.alpha_composite(bg, fg)
    if show:
        img.show()
    return img

def problem_to_image(p, show=False):
    pass