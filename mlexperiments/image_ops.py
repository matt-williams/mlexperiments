import math
import base64
import os
import inspect
import matplotlib
import numpy as np
from io import BytesIO
from PIL import Image, ImageFont, ImageDraw, ImageFilter
from IPython import display

def rgb_to_yuv(rgb):
    yuv = np.dot(rgb, [[0.29900, -0.16874,  0.50000],
                       [0.58700, -0.33126, -0.41869],
                       [0.11400,  0.50000, -0.08131]])
    yuv[:, :, 0] -= 0.5
    return yuv

def yuv_to_rgb(yuv):
    yuv += 0.5
    rgb = np.dot(yuv, [[ 1.0,                      1.0,                1.0],
                       [-0.000007154783816076815, -0.3441331386566162, 1.7720025777816772],
                       [ 1.4019975662231445,      -0.7141380310058594, 0.00001542569043522235]])
    rgb[:,:,0] -= 0.703744206526408
    rgb[:,:,1] += 0.5312106263403799
    rgb[:,:,2] -= 0.8894835468409035
    rgb = rgb.clip(0, 1)
    return rgb

def rgb_to_yuv_torch(rgb):
    Variable = type(rgb)
    yuv = Variable(rgb.data.new(*rgb.size()))
    yuv[:, 0, :, :] =  0.29900 * rgb[:, 0, :, :] +  0.58700 * rgb[:, 1, :, :] +  0.11400 * rgb[:, 2, :, :] - 0.5
    yuv[:, 1, :, :] = -0.16874 * rgb[:, 0, :, :] + -0.33126 * rgb[:, 1, :, :] +  0.50000 * rgb[:, 2, :, :]
    yuv[:, 2, :, :] =  0.50000 * rgb[:, 0, :, :] + -0.41869 * rgb[:, 1, :, :] + -0.08131 * rgb[:, 2, :, :]
    return yuv

def yuv_to_rgb_torch(yuv):
    Variable = type(yuv)
    rgb = Variable(yuv.data.new(*yuv.size()))
    yuv += 0.5
    rgb[:, 0, :, :] = yuv[:, 0, :, :] + -0.000007154783816076815 * yuv[:, 1, :, :] +  1.4019975662231445     * yuv[:, 2, :, :] - 0.703744206526408
    rgb[:, 1, :, :] = yuv[:, 0, :, :] + -0.3441331386566162      * yuv[:, 1, :, :] + -0.714138031005859      * yuv[:, 2, :, :] + 0.5312106263403799
    rgb[:, 2, :, :] = yuv[:, 0, :, :] +  1.7720025777816772      * yuv[:, 1, :, :] +  0.00001542569043522235 * yuv[:, 2, :, :] - 0.8894835468409035
    rgb = rgb.clamp(0, 1)
    return rgb

def uv_to_rgb_torch(x):
    return x.cat([x, x[:, 0:1]], dim=1)

def hsv_to_rgb(h, s, v):
    h -= math.floor(h)
    h = h * 6
    if h < 1:
        [r, g, b] = [1, h, 0]
    elif h < 2:
        [r, g, b] = [2 - h, 1, 0]
    elif h < 3:
        [r, g, b] = [0, 1, h - 2]
    elif h < 4:
        [r, g, b] = [0, 4 - h, 1]
    elif h < 5:
        [r, g, b] = [h - 4, 0, 1]
    else:
        [r, g, b] = [1, 0, 6 - h]
    return [((r - 0.5) * s + 0.5) * v, ((g - 0.5) * s + 0.5) * v, ((b - 0.5) * s + 0.5) * v]

hash_colorize = np.array([[0, 0, 0] if i == 0 else
                              hsv_to_rgb(((i + 1) % 17)/17,
                                         ((i + 1) % 3)/3 / 2 + 0.5,
                                         ((i + 1) % 5)/5 * 2/3 + 1/3)
                          for i in range(256)])


def vignette(shape=(60, 80), scale=(1, 1), power=(1, 1)):
    x = np.expand_dims(np.power(np.cos(np.arange(-(shape[1] // 2) + 0.5, shape[1] - shape[1] // 2 + 0.5) * math.pi / shape[1] * scale[1]), power[1]), axis=0)
    y = np.expand_dims(np.power(np.cos(np.arange(-(shape[0] // 2) + 0.5, shape[0] - shape[0] // 2 + 0.5) * math.pi / shape[0] * scale[0]), power[0]), axis=1)
    return x * y


def checkerboard(size, check_size):
    total_size = [(x + 2 * check_size - 1) // (2 * check_size) for x in size]
    total_image = np.kron([[1, 0] * total_size[1], [0, 1] * total_size[1]] * total_size[0], np.ones((check_size, check_size)))
    return total_image[:size[0], :size[1]]

def red_green_image(image):
    if len(image.shape) == 2:
        image = np.expand_dims(image, 0)
    image = np.clip(image, 0, 1)
    image = np.concatenate([image, 1-image, np.zeros_like(image)], axis=2 if image.shape[2] == 1 else 0)
    return image

def apply_image_mask(image, mask):
    if len(image.shape) == 2:
        image = np.expand_dims(image, 0)
    image = np.clip(image, 0, 1)
    if image.shape[0] == 1:
        image = red_green_image(image)
    image = image * mask + (0.5 * checkerboard(image.shape[1:], 10) + 0.25) * (1 - mask)
    return image

def apply_image_sigma(image, sigma, mask=None):
    sigma = np.clip(sigma, -10, 10) # avoid over-or-under-flow while displaying images
    sigma_mask = 1 - np.clip(np.exp(sigma), 0, 1)
    if mask:
        sigma_mask *= mask
    return apply_image_mask(image, sigma_mask)

def layer_to_image(layer, out_ch=1, num_cols=None, num_rows=None):
    in_ch = layer.shape[1]
    out_cells = math.ceil(in_ch / out_ch)
    if not num_cols and not num_rows:
        num_rows = 1
    if not num_cols:
        num_cols = math.ceil(out_cells / num_rows)
    if not num_rows:
        num_rows = math.ceil(out_cells / num_cols)
    zeros = layer[:, 0:1] * 0
    rows = []
    for y in range(num_rows):
        cells = []
        for x in range(num_cols):
            ii = x + y * num_cols
            if (ii + 1) * out_ch <= in_ch:
                cell = layer[:, ii * out_ch:(ii + 1) * out_ch]
            elif ii * out_ch < in_ch:
                cell = layer[:, ii * out_ch:]
            else:
                cell = zeros
            if cell.shape[1] < out_ch:
                cell_padding = zeros.expand(cell.shape[0], out_ch - cell.shape[1], *cell.shape[2:])
                cell = cell.cat([cell, cell_padding], dim=1)
            cells.append(cell)
        row = cells[0].cat(cells, dim=3)
        rows.append(row)
    return rows[0].cat(rows, dim=2)


def draw_outlined_text(image, position, text, font, outline):
    size = font.getsize(text)
    size = (size[0], size[1] * 2 * (text.rstrip().count('\n') + 1))
    alpha = Image.new('L', (size[0] + 2 * outline, size[1] + 2 * outline), 'black')
    ImageDraw.Draw(alpha).text((outline, outline), text, 'white', font)
    alpha = alpha.filter(ImageFilter.MaxFilter(2 * outline + 1))
    luma = Image.new('L', (size[0] + 2 * outline, size[1] + 2 * outline), 'black')
    ImageDraw.Draw(luma).text((outline, outline), text, 'white', font)
    image.paste(luma, (position[0] + outline, position[1] + outline), mask=alpha)

def animate_frames(frames, fps=10, scale=1):
    frames = [Image.fromarray((frame * 255.0).astype(np.uint8) if frame.dtype == np.float32 or frame.dtype == np.float64 else frame) if isinstance(frame, np.ndarray) else frame for frame in frames]
    data = BytesIO()
    frames[0].save(data, format='gif', save_all=True, append_images=frames[1:], loop=0, duration=1000 / fps)
    html = '<img width="{}" height="{}" src="data:image/gif;base64,{}" />'.format(frames[0].width * scale, frames[0].height * scale, base64.b64encode(data.getvalue()).decode())
    display.display(display.HTML(html))
    del data, frames, html

font = ImageFont.truetype('{}/mpl-data/fonts/ttf/DejaVuSans-Bold.ttf'.format(os.path.dirname(inspect.getfile(matplotlib))), size=16, encoding="unic")    

def sample_to_image(sample, images=[["color", "depth"], ["types", "instances"]], message_fmt=None):
    images = [images] if isinstance(images, str) else images
    images = [images] if isinstance(images[0], str) else images
    rows = []
    for row in images:
        cells = []
        for key in row:
            cell_sigma = None
            if ":" in key:
                key, diff_key = key.split(":", 1)
                cell = (sample[key] - sample[diff_key]) * 0.5 + 0.5
            if key.endswith("+mask+sigma"):
                key = key[:-len("+mask+sigma")]
                cell = apply_image_sigma(sample[key], sample[key + "_sigma"], mask=sample[key + "_mask"])
            elif key.endswith("+sigma"):
                key = key[:-len("+sigma")]
                cell = apply_image_sigma(sample[key], sample[key + "_sigma"])
            elif key.endswith("+mask"):
                key = key[:-len("+mask")]
                cell = apply_image_mask(sample[key], sample[key + "_mask"])
            else:
                cell = sample[key]
                if cell.dtype == np.float32 or cell.dtype == np.float64:
                    cell = np.clip(cell, 0, 1)
            cell = np.expand_dims(cell, 0) if len(cell.shape) < 3 else cell
            cell = np.transpose(cell, (1, 2, 0))
            if key.find("depth") != -1 and cell.shape[2] == 1:
                cell = red_green_image(cell)
            elif key.find("types") != -1 or key.find("instances") != -1:
                cell = hash_colorize[cell[:, :, 0]]
            cell = np.repeat(cell, 3, 2) if cell.shape[2] == 1 else cell
            cell = (cell * 255.0).astype(np.uint8) if cell.dtype == np.float32 or cell.dtype == np.float64 else cell
            cells.append(cell)
        rows.append(np.concatenate(cells, axis=1))
        del cells
    image = np.concatenate(rows, axis=0)
    image = Image.fromarray(image)
    del images, rows
    if message_fmt:
        message = message_fmt.format(**sample)
        draw_outlined_text(image, (1, 1), message, font, 2)
    return image

def display_frame(frame, scale=1):
    image = Image.fromarray((frame * 255.0).astype(np.uint8) if frame.dtype == np.float32 or frame.dtype == np.float64 else frame) if isinstance(frame, np.ndarray) else frame
    data = BytesIO()
    image.save(data, format='png')
    html = '<img width="{}" height="{}" src="data:image/gif;base64,{}" />'.format(image.width * scale, image.height * scale, base64.b64encode(data.getvalue()).decode())
    display.display(display.HTML(html))
    del image, data, html

def display_frames(frames, num_cols=1, scale=1):
    num_rows = (len(frames) + num_cols - 1) // num_cols
    cols = [frames[ii * num_rows : (ii + 1) * num_rows] for ii in range(num_cols)]
    cols = [np.concatenate(col, axis=0) for col in cols]
    max_height = max([col.shape[0] for col in cols])
    for ii, col in enumerate(cols):
        if col.shape[0] < max_height:
            cols[ii] = np.concatenate([col, np.ones((max_height - col.shape[0], col.shape[1], col.shape[2]), dtype=col.dtype)], axis=0)
    frame = np.concatenate(cols, axis=1)
    display_frame(frame, scale=scale)
    del cols, frame
    
def display_samples(samples, images=[["color", "depth"], ["types", "instances"]], message_fmt=None, num_cols=1, scale=1):
    samples = [samples] if not isinstance(samples, list) else samples
    frames = [sample_to_image(sample, images, message_fmt) for sample in samples]
    display_frames(frames, num_cols=num_cols, scale=scale)
    del samples, frames

def animate_samples(samples, images=[["color", "depth"], ["types", "instances"]], message_fmt=None, fps=10, scale=1):
    frames = []
    for sample in samples:
        frames.append(sample_to_image(sample, images=images, message_fmt=message_fmt))
    animate_frames(frames, fps, scale)

def torch_samples_to_np_samples(torch_samples):
    np_samples = []
    for key in torch_samples:
        value = torch_samples[key]
        try:
            if hasattr(value, "data"):
                value = value.data
        except RuntimeError:
            # Torch Tensors throw a RuntimeError if you even ask if they have a data property
            pass
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        for ii in range(value.shape[0] if hasattr(value, "shape") else len(value)):
            if ii >= len(np_samples):
                np_samples.append({})
            np_samples[ii][key] = value[ii]
        del value
    return np_samples