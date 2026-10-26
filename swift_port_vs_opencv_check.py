# Python port of the Swift algorithms (same math), compared against OpenCV.
import numpy as np, cv2, math

def gray(img):
    r, g, b = [img[..., i].astype(np.int64) for i in range(3)]
    return ((r * 9798 + g * 19235 + b * 3735 + 16384) >> 15).astype(np.uint8)  # OpenCV 5 in this sandbox

def crop(img, tol=7):
    m = gray(img) > tol
    if not m.any(): return img
    ys, xs = np.where(m.any(1))[0], np.where(m.any(0))[0]
    return img[ys.min():ys.max()+1, xs.min():xs.max()+1]

def table(dn, sn, area=False):
    scale, inv = sn / dn, dn / sn
    i0, i1, c0 = [], [], []
    for d in range(dn):
        if area:
            s = math.floor(d * scale); f = np.float32((d + 1) - (s + 1) * inv)
            f = np.float32(0) if f <= 0 else f - np.float32(math.floor(f))
        else:
            fx = np.float32((d + 0.5) * scale - 0.5); s = math.floor(fx); f = fx - np.float32(s)
        if s < 0: f, s = np.float32(0), 0
        if s >= sn - 1: f, s = np.float32(0), sn - 1
        w0 = int(np.rint(np.float32((np.float32(1) - f) * np.float32(2048))))
        i0.append(s); i1.append(min(s + 1, sn - 1)); c0.append(w0)
    return np.array(i0), np.array(i1), np.array(c0), 2048 - np.array(c0)

def resize_linear(src, dw, dh, area=False):
    sh, sw = src.shape[:2]
    if (sw, sh) == (dw, dh): return src
    x0, x1, a0, a1 = table(dw, sw, area); y0, y1, b0, b1 = table(dh, sh, area)
    s = src.astype(np.int64)
    H0 = s[y0][:, x0] * a0[None, :, None] + s[y0][:, x1] * a1[None, :, None]
    H1 = s[y1][:, x0] * a0[None, :, None] + s[y1][:, x1] * a1[None, :, None]
    B0, B1 = b0[:, None, None], b1[:, None, None]
    v = (((B0 * (H0 >> 4)) >> 16) + ((B1 * (H1 >> 4)) >> 16) + 2) >> 2
    return np.clip(v, 0, 255).astype(np.uint8)

def r101(i, n):
    while i < 0 or i >= n:
        if i < 0: i = -i
        if i >= n: i = 2 * n - 2 - i
    return i

def gblur(img, sigma):
    n = int(np.rint(sigma * 6 + 1)) | 1
    x = np.arange(n) - (n - 1) / 2
    k = np.exp(-x * x / (2 * sigma * sigma)); k = (k / k.sum()).astype(np.float32)
    h, w = img.shape[:2]; r = n // 2
    ix = np.array([r101(i - r, w) for i in range(w + 2 * r)]); iy = np.array([r101(i - r, h) for i in range(h + 2 * r)])
    f = img.astype(np.float32)
    P = f[:, ix]; Hh = sum(P[:, j:j + w] * k[j] for j in range(n))
    P = Hh[iy]; V = sum(P[j:j + h] * k[j] for j in range(n))
    return np.clip(np.rint(V), 0, 255).astype(np.uint8)

def ben(raw):
    c = crop(raw); r = resize_linear(c, 380, 380); b = gblur(r, 380 / 30)
    return np.clip(4 * r.astype(int) - 4 * b.astype(int) + 128, 0, 255).astype(np.uint8)

def ben_cv(raw):
    c = crop(raw); r = cv2.resize(c, (380, 380)); b = cv2.GaussianBlur(r, (0, 0), 380 / 30)
    return cv2.addWeighted(r, 4, b, -4, 128)

def area_tab(dn, sn):
    scale = sn / dn; tabs = []
    for d in range(dn):
        fs1 = d * scale; fs2 = fs1 + scale; cell = min(scale, sn - fs1)
        s1 = math.ceil(fs1); s2 = math.floor(fs2); s2 = min(s2, sn - 1); s1 = min(s1, s2); t = []
        if s1 - fs1 > 1e-3: t.append((s1 - 1, (s1 - fs1) / cell))
        for s in range(s1, s2): t.append((s, 1 / cell))
        if fs2 - s2 > 1e-3: t.append((s2, min(min(fs2 - s2, 1), cell) / cell))
        tabs.append(t)
    return tabs

def resize_area(src, dw, dh):
    sh, sw = src.shape[:2]
    if sw < dw or sh < dh: return resize_linear(src, dw, dh, area=True)
    W = np.zeros((dw, sw), np.float32)
    for d, t in enumerate(area_tab(dw, sw)):
        for s, w in t: W[d, s] += w
    Hm = np.zeros((dh, sh), np.float32)
    for d, t in enumerate(area_tab(dh, sh)):
        for s, w in t: Hm[d, s] += w
    f = src.astype(np.float32)
    h1 = np.einsum('ds,hsc->hdc', W, f); v = np.einsum('ds,swc->dwc', Hm, h1)
    return np.clip(np.rint(v), 0, 255).astype(np.uint8)

rng = np.random.default_rng(0)
def fake_fundus(h, w):
    yy, xx = np.mgrid[0:h, 0:w]; cy, cx, R = h / 2, w / 2, min(h, w) * 0.46
    base = np.stack([150 + 40 * np.sin(xx / 37.0), 70 + 30 * np.cos(yy / 23.0), 30 + 10 * np.sin((xx + yy) / 11.0)], -1)
    base += rng.normal(0, 6, base.shape)
    for _ in range(40):
        y, x = rng.integers(0, h), rng.integers(0, w); base[max(0,y-3):y+3, max(0,x-3):x+3] *= 0.4
    img = np.clip(base, 0, 255).astype(np.uint8); img[(yy - cy) ** 2 + (xx - cx) ** 2 > R ** 2] = rng.integers(0, 5)
    return img

print("cv2", cv2.__version__)
for (h, w) in [(1050, 1050), (1736, 2416), (2848, 4288), (512, 512), (300, 330)]:
    raw = fake_fundus(h, w)
    c = crop(raw); g_cv = cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY)
    gray_ok = np.array_equal(g_cv, gray(raw))
    rs = np.abs(resize_linear(c, 380, 380).astype(int) - cv2.resize(c, (380, 380)).astype(int))
    r = cv2.resize(c, (380, 380))
    bl = np.abs(gblur(r, 380 / 30).astype(int) - cv2.GaussianBlur(r, (0, 0), 380 / 30).astype(int))
    bd = np.abs(ben(raw).astype(int) - ben_cv(raw).astype(int))
    ar = np.abs(resize_area(c, 512, 512).astype(int) - cv2.resize(c, (512, 512), interpolation=cv2.INTER_AREA).astype(int))
    print(f"{w}x{h}: gray exact={gray_ok} | resize max={rs.max()} mean={rs.mean():.4f} | blur max={bl.max()} mean={bl.mean():.4f} "
          f"| BenGraham max={bd.max()} mean={bd.mean():.4f} | area512 max={ar.max()} mean={ar.mean():.4f}")

# Laplacian ksize=3 kernel check
g = gray(fake_fundus(200, 200)); lap_cv = cv2.Laplacian(g, cv2.CV_64F, ksize=3)
gp = np.pad(g.astype(float), 1, mode='reflect')
mine = 2 * (gp[:-2, :-2] + gp[:-2, 2:] + gp[2:, :-2] + gp[2:, 2:]) - 8 * gp[1:-1, 1:-1]
print("Laplacian ksize=3 diag kernel matches OpenCV:", np.allclose(lap_cv, mine))
