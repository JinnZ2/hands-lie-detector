"""
Image input for the attention-as-sensor probe. Stdlib only.

The standard library has no JPEG or HEIC decoder. This module decodes two
formats and refuses the rest:

    PNG   8-bit, non-interlaced, colour types 0 (gray), 2 (RGB), 3 (palette),
          4 (gray + alpha), 6 (RGBA). zlib inflate, filters 0-4, CRC checked.
          Alpha is dropped (logged in bounds). 16-bit and interlaced PNG are
          refused, not converted.
    PPM   P6 (binary) and P3 (ASCII), maxval up to 255.

A JPEG or HEIC frame is refused by its magic bytes, with the alternative named:
coordinates-only mode, where regions are logged by pixel coordinates and the
physics layer reads UNMEASURED. `Frame.input_mode` states which mode ran.
"""

import hashlib
import struct
import zlib
from dataclasses import dataclass, field


class UnsupportedImage(ValueError):
    pass


@dataclass
class Frame:
    width: int
    height: int
    rgb: list            # row-major list of (r, g, b) tuples, 0-255; empty in coordinates mode
    sha256: str          # of the file bytes as supplied (never of a re-encoding)
    input_mode: str      # "decoded:PNG" | "decoded:PPM" | "coordinates_only"
    bounds: dict = field(default_factory=dict)

    def pixel(self, x, y):
        return self.rgb[y * self.width + x]

    @property
    def decoded(self):
        return self.input_mode.startswith("decoded:")


_PNG_SIG = b"\x89PNG\r\n\x1a\n"
_REFUSED = {
    b"\xff\xd8\xff": "JPEG",
}


def _sniff(data):
    for magic, name in _REFUSED.items():
        if data.startswith(magic):
            return name
    if len(data) > 12 and data[4:8] == b"ftyp" and data[8:12] in (
            b"heic", b"heix", b"hevc", b"heim", b"heis", b"mif1", b"msf1", b"avif"):
        return "HEIC/HEIF"
    return None


def _paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def decode_png(data):
    if not data.startswith(_PNG_SIG):
        raise UnsupportedImage("not a PNG")
    pos, idat, plte, ihdr = 8, [], None, None
    while pos < len(data):
        if pos + 8 > len(data):
            raise UnsupportedImage("PNG truncated")
        (length,), ctype = struct.unpack(">I", data[pos:pos + 4]), data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        (crc,) = struct.unpack(">I", data[pos + 8 + length:pos + 12 + length])
        if zlib.crc32(ctype + body) & 0xFFFFFFFF != crc:
            raise UnsupportedImage(f"PNG chunk {ctype!r}: CRC mismatch")
        if ctype == b"IHDR":
            ihdr = struct.unpack(">IIBBBBB", body)
        elif ctype == b"PLTE":
            plte = [tuple(body[i:i + 3]) for i in range(0, len(body), 3)]
        elif ctype == b"IDAT":
            idat.append(body)
        elif ctype == b"IEND":
            break
        pos += 12 + length
    if ihdr is None:
        raise UnsupportedImage("PNG has no IHDR")
    w, h, depth, ctype, _comp, _filt, interlace = ihdr
    if depth != 8:
        raise UnsupportedImage(f"PNG bit depth {depth}: only 8-bit is decoded")
    if interlace:
        raise UnsupportedImage("interlaced PNG is not decoded")
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(ctype)
    if channels is None:
        raise UnsupportedImage(f"PNG colour type {ctype} not decoded")
    if ctype == 3 and not plte:
        raise UnsupportedImage("palette PNG without PLTE")
    raw = zlib.decompress(b"".join(idat))
    stride = w * channels
    out, prev = [], bytearray(stride)
    for y in range(h):
        base = y * (stride + 1)
        f, line = raw[base], bytearray(raw[base + 1:base + 1 + stride])
        for i in range(stride):
            a = line[i - channels] if i >= channels else 0
            b = prev[i]
            c = prev[i - channels] if i >= channels else 0
            if f == 1:
                line[i] = (line[i] + a) & 0xFF
            elif f == 2:
                line[i] = (line[i] + b) & 0xFF
            elif f == 3:
                line[i] = (line[i] + ((a + b) >> 1)) & 0xFF
            elif f == 4:
                line[i] = (line[i] + _paeth(a, b, c)) & 0xFF
            elif f != 0:
                raise UnsupportedImage(f"PNG filter type {f} unknown")
        for x in range(w):
            px = line[x * channels:(x + 1) * channels]
            if ctype in (0, 4):
                out.append((px[0], px[0], px[0]))
            elif ctype == 3:
                out.append(plte[px[0]])
            else:
                out.append((px[0], px[1], px[2]))
        prev = line
    return w, h, out, {"png_colour_type": ctype, "alpha_dropped": ctype in (4, 6)}


def decode_ppm(data):
    magic = data[:2]
    if magic not in (b"P6", b"P3"):
        raise UnsupportedImage("not a PPM (P6/P3)")
    tokens, pos = [], 2
    while len(tokens) < 3:
        while pos < len(data) and data[pos:pos + 1].isspace():
            pos += 1
        if data[pos:pos + 1] == b"#":
            while pos < len(data) and data[pos:pos + 1] not in (b"\n", b"\r"):
                pos += 1
            continue
        start = pos
        while pos < len(data) and not data[pos:pos + 1].isspace():
            pos += 1
        tokens.append(int(data[start:pos]))
    w, h, maxval = tokens
    if not 0 < maxval <= 255:
        raise UnsupportedImage(f"PPM maxval {maxval}: only up to 255 is decoded")
    scale = 255.0 / maxval
    if magic == b"P6":
        body = data[pos + 1:pos + 1 + 3 * w * h]
        vals = list(body)
    else:
        vals = [int(t) for t in data[pos:].split()][:3 * w * h]
    if len(vals) < 3 * w * h:
        raise UnsupportedImage("PPM truncated")
    vals = [round(v * scale) for v in vals]
    rgb = [tuple(vals[i:i + 3]) for i in range(0, 3 * w * h, 3)]
    return w, h, rgb, {"ppm_maxval": maxval}


def load(path):
    with open(path, "rb") as fh:
        data = fh.read()
    sha = hashlib.sha256(data).hexdigest()
    refused = _sniff(data)
    if refused:
        raise UnsupportedImage(
            f"{refused} is not decoded (standard library has no {refused} decoder). "
            f"Convert to PNG or PPM, or run coordinates-only mode (--size W H), where "
            f"the physics layer reads UNMEASURED")
    if data.startswith(_PNG_SIG):
        w, h, rgb, extra = decode_png(data)
        mode = "decoded:PNG"
    elif data[:2] in (b"P6", b"P3"):
        w, h, rgb, extra = decode_ppm(data)
        mode = "decoded:PPM"
    else:
        raise UnsupportedImage("unrecognised image format: give PNG or PPM, or use --size W H")
    return Frame(w, h, rgb, sha, mode, {"decoded_size": [w, h], **extra})


def coordinates_only(width, height, sha256=None):
    """No pixels: regions are logged by coordinates and the physics layer is UNMEASURED."""
    return Frame(width, height, [], sha256, "coordinates_only", {"declared_size": [width, height]})


def small(frame, max_side=128):
    """--small: integer block-mean downsample so the longer side is <= max_side."""
    if not frame.decoded:
        return frame, 1
    f = -(-max(frame.width, frame.height) // max_side)
    if f <= 1:
        return frame, 1
    w, h = frame.width // f, frame.height // f
    rgb = []
    for by in range(h):
        for bx in range(w):
            acc = [0, 0, 0]
            for y in range(by * f, by * f + f):
                for x in range(bx * f, bx * f + f):
                    p = frame.pixel(x, y)
                    acc[0] += p[0]; acc[1] += p[1]; acc[2] += p[2]
            n = f * f
            rgb.append((acc[0] / n, acc[1] / n, acc[2] / n))
    b = dict(frame.bounds, small_factor=f, small_size=[w, h],
             small_note="block mean; dropped the last width % f columns and height % f rows")
    return Frame(w, h, rgb, frame.sha256, frame.input_mode, b), f
