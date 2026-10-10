# -*- coding: utf-8 -*-
"""Notebook ADA v1 — phần 2: kiến trúc, giao thức đánh giá, bộ máy huấn luyện, chạy thí nghiệm chính."""

CELLS = []


def md(s):
    CELLS.append(("markdown", s))


def code(s):
    CELLS.append(("code", s))


# =============================================================================
# 5. KIẾN TRÚC
# =============================================================================
md(r'''
## 5. Kiến trúc mạng / Network architecture

| Khối | Cấu trúc | Đầu ra |
|---|---|---|
| Front-end (trong mô hình, chạy trên GPU) | log-Mel `n_mels`=64, `n_fft`=2048, hop 512 @ 16 kHz, `f_max` = băng thông chung (mục 3.2); SpecAugment **trước** chuẩn hoá; chuẩn hoá theo từng mẫu | 1×64×63 |
| $G_f$ — trích đặc trưng | 4 × [Conv3×3–BN–ReLU ×2, Squeeze-Excitation, MaxPool, Dropout2d 0.1], kênh 32/64/128/256 (giống `LeakCNN2D` v7) → GAP ⊕ GMP | 512 |
| Bottleneck | FC 512→256, BN, ReLU, Dropout 0.3 → **đặc trưng $f$ dùng để căn chỉnh miền** | 256 |
| $G_y$ — dự đoán nhãn | FC 256→1 (sigmoid) | xác suất rò rỉ |
| $G_d$ — phân biệt miền (DANN, RCST) | **GRL($-\lambda_p$)** → FC 256→1024–ReLU–Dropout 0.5 → FC 1024→1024–ReLU–Dropout 0.5 → FC 1024→1, **lr ×10** (theo bản cài đặt gốc; v1 dùng 256 / ×1 — quá yếu) | nguồn / đích |
| $G_d$ — CDAN(+E) | GRL → **$f \otimes g$** (256 × 2 = 512, $g=[1-p, p]$ tách gradient) → cùng MLP | nguồn / đích |
| Deep CORAL / MK-MMD / MCC | Không có $G_d$; thêm $\lambda_p \cdot$ CORAL($f_s, f_t$), MMD($f_s, f_t$) hoặc MCC(dự đoán đích) | — |
| Joint-BN (đối chứng cơ chế) | Lô trộn nguồn + đích qua cùng một lượt truyền, **không** có hàm căn chỉnh nào | — |
| **RCST (đề xuất)** | DANN + nhãn giả **mức bản ghi** (trung bình xác suất các frame, chọn cân bằng theo thứ hạng, tỉ lệ tăng dần) + phạt **không nhất quán giữa các frame cùng bản ghi**; lô đích = 8 bản ghi × 8 frame | — |

Dùng **cùng một $G_f$ + $G_y$** cho mọi phương pháp → khác biệt kết quả chỉ đến từ mục tiêu căn chỉnh miền.
''')

code(r'''
# =========================================================
# 5.1 Front-end log-Mel + SpecAugment (từ v7; [V2] dải f_min-f_max = dải chung sau hoà hợp chuỗi thu)
# =========================================================
def _safe_norm(m):
    m = torch.nan_to_num(m, nan=0.0, posinf=0.0, neginf=-80.0)
    mu = m.mean(dim=(2, 3), keepdim=True)
    sd = m.std(dim=(2, 3), keepdim=True).clamp_min(1e-3)
    return torch.nan_to_num((m - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)

class SpecAugment(nn.Module):
    # Che dải tần / khoảng thời gian, MỖI MẪU một mặt nạ, giá trị che = mức im lặng của mẫu, chỉ khi train.
    def __init__(self, f=None, t=None, n=2):
        super().__init__()
        self.f = cfg.specaug_freq_mask if f is None else int(f)
        self.t = cfg.specaug_time_mask if t is None else int(t)
        self.n = int(n)
        self.generator = None
    def _band(self, B, L, W, device):
        g = self.generator
        w = torch.randint(0, W + 1, (B,), generator=g)
        s0 = (torch.rand(B, generator=g) * (L - w).clamp_min(1).float()).long()
        ar = torch.arange(L)
        return ((ar[None] >= s0[:, None]) & (ar[None] < (s0 + w)[:, None])).to(device)
    def forward(self, s):
        if not self.training or (self.f <= 0 and self.t <= 0):
            return s
        B, C, Fq, Tm = s.shape
        fill = s.amin(dim=(2, 3), keepdim=True).detach()
        mask = torch.zeros(B, 1, Fq, Tm, dtype=torch.bool, device=s.device)
        for _ in range(self.n):
            if self.f > 0 and Fq > self.f:
                mask = mask | self._band(B, Fq, self.f, s.device)[:, None, :, None]
            if self.t > 0 and Tm > self.t:
                mask = mask | self._band(B, Tm, self.t, s.device)[:, None, None, :]
        return torch.where(mask, fill, s)

class LogMelFrontend(nn.Module):
    def __init__(self, specaug=None):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.target_sr, n_fft=cfg.n_fft, hop_length=cfg.hop_length, n_mels=cfg.n_mels,
            f_min=F_MIN_EFF, f_max=F_MAX_EFF, power=2.0, center=True)
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)
        self.specaug = specaug
    def forward(self, x):
        with amp_autocast(x.device, False):                 # front-end luôn fp32 (tránh tràn fp16)
            s = self.to_db(self.mel(x.float())).unsqueeze(1)
        if self.specaug is not None:
            s = self.specaug(s)
        return _safe_norm(s)

with torch.no_grad():
    _s = LogMelFrontend()(torch.from_numpy(np.asarray(X_ALL[:2])).float())
SPEC_SHAPE = tuple(_s.shape[1:])
print("log-Mel:", SPEC_SHAPE, f"| {F_MIN_EFF:.0f}-{F_MAX_EFF:.0f} Hz")
''')

code(r'''
# =========================================================
# 5.2 G_f + bottleneck + G_y, GRL, bộ phân biệt miền, CDAN, CORAL, MK-MMD
# =========================================================
class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(ch, max(4, ch // r)), nn.ReLU(inplace=True),
                                nn.Linear(max(4, ch // r), ch), nn.Sigmoid())
    def forward(self, x):
        return x * self.fc(x.mean(dim=(2, 3)))[:, :, None, None]

class ConvBlock2D(nn.Module):
    def __init__(self, cin, cout, pool=(2, 2), p_drop=0.1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
        self.se = SEBlock(cout)
        self.pool = nn.MaxPool2d(pool)
        self.drop = nn.Dropout2d(p_drop)
    def forward(self, x):
        return self.drop(self.pool(self.se(self.body(x))))

def gap_gmp(h):
    dims = tuple(range(2, h.dim()))
    return torch.cat([h.mean(dim=dims), h.amax(dim=dims)], dim=1)

class LeakNet(nn.Module):
    """G_f (front-end + CNN) -> bottleneck f -> G_y. Dùng chung cho MỌI phương pháp."""
    def __init__(self, backbone=None):
        super().__init__()
        self.backbone = backbone or cfg.backbone
        if self.backbone == "cnn2d":
            self.specaug = SpecAugment()
            self.front = LogMelFrontend(specaug=self.specaug)
            blocks, cin = [], 1
            for i, w in enumerate(cfg.widths):
                blocks.append(ConvBlock2D(cin, w, pool=(2, 2) if i < len(cfg.widths) - 1 else (2, 1)))
                cin = w
            self.blocks = nn.Sequential(*blocks)
            d_trunk = 2 * cin
        else:
            c1, c2, c3, c4 = [2 * w for w in cfg.widths]
            self.net1d = nn.Sequential(
                nn.Conv1d(1, c1, 80, stride=4, padding=38, bias=False), nn.BatchNorm1d(c1), nn.ReLU(True),
                nn.MaxPool1d(4),
                nn.Conv1d(c1, c2, 3, padding=1, bias=False), nn.BatchNorm1d(c2), nn.ReLU(True), nn.MaxPool1d(4),
                nn.Conv1d(c2, c3, 3, padding=1, bias=False), nn.BatchNorm1d(c3), nn.ReLU(True), nn.MaxPool1d(4),
                nn.Conv1d(c3, c4, 3, padding=1, bias=False), nn.BatchNorm1d(c4), nn.ReLU(True))
            d_trunk = 2 * c4
        self.bottleneck = nn.Sequential(nn.Linear(d_trunk, cfg.bottleneck_dim),
                                        nn.BatchNorm1d(cfg.bottleneck_dim), nn.ReLU(inplace=True),
                                        nn.Dropout(0.3))
        self.cls = nn.Linear(cfg.bottleneck_dim, 1)

    def spec(self, x):
        return self.front(x)
    def trunk_from_spec(self, s):
        return gap_gmp(self.blocks(s))
    def trunk(self, x):
        if self.backbone == "cnn2d":
            return self.trunk_from_spec(self.front(x))
        x = x.float()
        if HARMONISED:                                         # cùng dải chung với front-end phổ
            with amp_autocast(x.device, False):
                if F_MAX_EFF < cfg.target_sr / 2 - 1:
                    x = torchaudio.functional.lowpass_biquad(x, cfg.target_sr, F_MAX_EFF)
                if F_MIN_EFF > 25:
                    x = torchaudio.functional.highpass_biquad(x, cfg.target_sr, F_MIN_EFF)
        x = (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True).clamp_min(1e-4)
        return gap_gmp(self.net1d(x.unsqueeze(1)))
    def head(self, z):
        f = self.bottleneck(z)
        return self.cls(f).squeeze(-1), f
    def forward(self, x, return_feat=False):
        logit, f = self.head(self.trunk(x))
        return (logit, f) if return_feat else logit


class GradReverse(torch.autograd.Function):
    # Lớp đảo gradient (Ganin & Lempitsky, 2015): xuôi = đồng nhất, ngược = -coeff * gradient.
    @staticmethod
    def forward(ctx, x, coeff):
        ctx.coeff = float(coeff)
        return x.view_as(x)
    @staticmethod
    def backward(ctx, g):
        return -ctx.coeff * g, None

class DomainDisc(nn.Module):
    def __init__(self, din, hidden=None, p=None):
        super().__init__()
        h = hidden or cfg.disc_hidden; p = cfg.disc_dropout if p is None else p
        self.net = nn.Sequential(nn.Linear(din, h), nn.ReLU(inplace=True), nn.Dropout(p),
                                 nn.Linear(h, h), nn.ReLU(inplace=True), nn.Dropout(p), nn.Linear(h, 1))
    def forward(self, h, coeff=1.0):
        return self.net(GradReverse.apply(h, coeff)).squeeze(-1)

def cdan_input(f, logit):
    # Ánh xạ đa tuyến f ⊗ g (Long et al., 2018); g = [1-p, p] được TÁCH gradient như bản gốc.
    p = torch.sigmoid(logit.float()).detach()
    g = torch.stack([1 - p, p], dim=1).to(f.dtype)
    return torch.bmm(g.unsqueeze(2), f.unsqueeze(1)).flatten(1)

def binary_entropy(logit):
    p = torch.sigmoid(logit.float()).clamp(1e-6, 1 - 1e-6)
    return -(p * torch.log(p) + (1 - p) * torch.log(1 - p))

def entropy_weights(logit_s, logit_t):
    # CDAN+E: w = 1 + exp(-H), chuẩn hoá RIÊNG trong nửa nguồn và nửa đích (mỗi miền tổng trọng số = 1).
    ws = 1 + torch.exp(-binary_entropy(logit_s)); wt = 1 + torch.exp(-binary_entropy(logit_t))
    return torch.cat([ws / ws.sum(), wt / wt.sum()]).detach()

def coral_loss(fs, ft):
    # Deep CORAL (Sun & Saenko, 2016): ||C_s - C_t||_F^2 / (4 d^2)
    fs, ft = fs.float(), ft.float(); d = fs.size(1)
    def cov(x):
        x = x - x.mean(0, keepdim=True)
        return x.t() @ x / max(x.size(0) - 1, 1)
    return ((cov(fs) - cov(ft)) ** 2).sum() / (4 * d * d)

def mmd_loss(fs, ft, mults=(0.25, 0.5, 1.0, 2.0, 4.0)):
    # MK-MMD (Long et al., 2015), ước lượng chệch, băng thông theo trung bình khoảng cách.
    x = torch.cat([fs, ft]).float(); n = x.size(0); ns = fs.size(0)
    d2 = torch.cdist(x, x).pow(2)
    bw = d2.detach().sum() / max(n * n - n, 1)
    K = sum(torch.exp(-d2 / (bw * m + 1e-9)) for m in mults)
    return K[:ns, :ns].mean() + K[ns:, ns:].mean() - 2 * K[:ns, ns:].mean()

def mcc_loss(logit_t, T=None):
    # Minimum Class Confusion (Jin et al., ECCV 2020) cho bài toán 2 lớp: giảm "nhầm lẫn lớp" giữa các mẫu
    # đích (ma trận tương quan dự đoán có trọng số entropy), không cần căn chỉnh phân bố.
    T = cfg.mcc_temp if T is None else T
    p = torch.sigmoid(logit_t.float() / T)
    P = torch.stack([1 - p, p], 1)
    H = -(P * torch.log(P + 1e-6)).sum(1)
    w = 1 + torch.exp(-H)
    w = (len(w) * w / w.sum()).detach()
    C = (P * w[:, None]).t() @ P
    C = C / C.sum(1, keepdim=True).clamp_min(1e-6)
    return (C.sum() - torch.trace(C)) / 2

def record_consistency(logit_t, grp, n_groups):
    # [V2-RCST] phương sai xác suất giữa các frame CỦA CÙNG MỘT BẢN GHI trong lô (mọi frame chung một nhãn).
    p = torch.sigmoid(logit_t.float())
    cnt = torch.zeros(n_groups, device=p.device).index_add_(0, grp, torch.ones_like(p)).clamp_min(1)
    m = torch.zeros(n_groups, device=p.device).index_add_(0, grp, p) / cnt
    return 4 * ((p - m[grp]) ** 2).mean()                  # x4 -> thang [0, 1]

# ---- Kiểm thử đơn vị / unit tests ----------------------------------------------------
_x = torch.randn(4, 3, requires_grad=True)
GradReverse.apply(_x, 0.5).sum().backward()
assert torch.allclose(_x.grad, torch.full_like(_x, -0.5)), "GRL phải nhân gradient với -coeff"
_a = torch.randn(64, 16)
assert float(coral_loss(_a, _a)) < 1e-10 and float(mmd_loss(_a, _a)) < 1e-6
assert float(coral_loss(_a, 3 * _a + 1)) > 0 and float(mmd_loss(_a, _a + 2)) > 0
_net = LeakNet(); _lg, _f = _net(torch.from_numpy(np.asarray(X_ALL[:3])).float(), return_feat=True)
assert _lg.shape == (3,) and _f.shape == (3, cfg.bottleneck_dim)
assert cdan_input(_f, _lg).shape == (3, 2 * cfg.bottleneck_dim)
_w = entropy_weights(_lg[:2], _lg[2:])
assert abs(float(_w.sum()) - 2.0) < 1e-5
assert float(mcc_loss(torch.tensor([9., 9., -9., -9.]))) < float(mcc_loss(torch.zeros(4)))   # tự tin -> ít nhầm lẫn
_g = torch.tensor([0, 0, 1, 1])
assert float(record_consistency(torch.tensor([2., 2., -3., -3.]), _g, 2)) < 1e-8
assert float(record_consistency(torch.tensor([5., -5., 0., 0.]), _g, 2)) > 0.1
print("✓ GRL, CORAL, MMD, CDAN, trọng số entropy, MCC, nhất quán bản ghi: kiểm thử đơn vị đạt.")
''')

code(r'''
# =========================================================
# 5.3 Bảng tham số + Hình: sơ đồ kiến trúc / Parameter table and architecture diagram
# =========================================================
def n_params(m):
    return int(sum(p.numel() for p in m.parameters()))

_net = LeakNet()
_trunk = n_params(_net.blocks) if cfg.backbone == "cnn2d" else n_params(_net.net1d)
arch = pd.DataFrame([
    dict(component="G_f feature extractor (CNN)", params=_trunk, output=f"{2 * cfg.widths[-1]}-d (GAP⊕GMP)"),
    dict(component="Bottleneck (FC-BN-ReLU-Dropout)", params=n_params(_net.bottleneck), output=f"{cfg.bottleneck_dim}-d f"),
    dict(component="G_y label predictor", params=n_params(_net.cls), output="1 logit"),
    dict(component="G_d domain discriminator (DANN)", params=n_params(DomainDisc(cfg.bottleneck_dim)),
         output="1 logit (source/target)"),
    dict(component="G_d domain discriminator (CDAN, f⊗g)", params=n_params(DomainDisc(2 * cfg.bottleneck_dim)),
         output="1 logit (source/target)"),
])
arch["params_M"] = (arch.params / 1e6).round(3)
save_table(arch, "architecture_parameters",
           f"Network components and trainable parameters (backbone '{cfg.backbone}', input {cfg.frame_sec:g}-s frame "
           f"at {cfg.target_sr} Hz, log-Mel {SPEC_SHAPE[1]}x{SPEC_SHAPE[2]}).", 3)
del _net

def _box(ax, xy, w, h, text, fc, ls="-"):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec="#333333", lw=1.1, ls=ls))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=8.2)

def _arrow(ax, a, b, color="#333333", ls="-"):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2, ls=ls))

fig, ax = plt.subplots(figsize=(15.5, 4.6))
ax.set_xlim(0, 15.3); ax.set_ylim(0, 5); ax.axis("off")
_box(ax, (0.1, 2.0), 1.6, 1.0, f"Waveform\n{cfg.frame_sec:g} s @ {cfg.target_sr // 1000} kHz\nsource + target", "#EEEEEE")
_box(ax, (2.1, 2.0), 2.5, 1.0, f"log-Mel front-end\n{SPEC_SHAPE[1]}×{SPEC_SHAPE[2]}, 0–{F_MAX_EFF / 1000:.1f} kHz\nSpecAugment + per-sample norm", "#DCE9F2")
_box(ax, (5.0, 2.0), 2.6, 1.0, "Feature extractor $G_f$\n4×[Conv-BN-ReLU×2, SE, Pool]\nGAP ⊕ GMP", "#DCE9F2")
_box(ax, (8.0, 2.0), 1.8, 1.0, f"Bottleneck\nFC-BN-ReLU-Drop\nf ∈ R^{cfg.bottleneck_dim}", "#DCE9F2")
_box(ax, (10.5, 3.6), 2.0, 0.9, "Label predictor $G_y$\nFC → sigmoid", "#F6E0D9")
_box(ax, (13.0, 3.6), 2.1, 0.9, "$\\mathcal{L}_y$: BCE\n(source labels)", "#FFFFFF")
_box(ax, (10.2, 0.4), 1.0, 0.9, "GRL\n$-\\lambda_p$", "#F3E3C3")
_box(ax, (11.5, 0.4), 1.5, 0.9, "CDAN: f ⊗ g\n(optional)", "#FFFFFF", ls="--")
_box(ax, (13.3, 0.4), 1.9, 0.9, f"Domain disc. $G_d$\nMLP {cfg.disc_hidden}-{cfg.disc_hidden}-1", "#E4DDF0")
_arrow(ax, (1.7, 2.5), (2.1, 2.5)); _arrow(ax, (4.6, 2.5), (5.0, 2.5)); _arrow(ax, (7.6, 2.5), (8.0, 2.5))
_arrow(ax, (9.8, 2.7), (10.5, 4.0)); _arrow(ax, (12.5, 4.05), (13.0, 4.05))
_arrow(ax, (9.8, 2.3), (10.2, 0.9)); _arrow(ax, (11.2, 0.85), (11.5, 0.85)); _arrow(ax, (13.0, 0.85), (13.3, 0.85))
_arrow(ax, (11.5, 3.6), (12.25, 1.3), color="#999999", ls="--")
ax.text(14.25, 0.12, "$\\mathcal{L}_d$: source vs target", ha="center", fontsize=8)
ax.text(8.9, 3.3, "Deep CORAL / MK-MMD act on f directly", ha="center", fontsize=7.5, color="#555555")
ax.text(9.75, 1.15, "reversed gradient\n$-\\lambda_p\\,\\partial\\mathcal{L}_d/\\partial f$", ha="right",
        fontsize=7.5, color="#7a5a10")
savefig(fig, "architecture_dann",
        "Adversarial domain adaptation network: shared feature extractor G_f and bottleneck f, label predictor G_y "
        "trained on labelled source frames, and domain discriminator G_d connected through a gradient reversal layer "
        "(GRL). CDAN conditions G_d on the multilinear map f ⊗ g of features and class probabilities.")
''')

# =============================================================================
# 6. GIAO THỨC ĐÁNH GIÁ
# =============================================================================
md(r'''
## 6. Bài toán chuyển miền và giao thức đánh giá / Transfer tasks and evaluation protocol

Với mỗi bài toán `nguồn → đích` và mỗi seed:

| Tập | Gồm | Dùng để |
|---|---|---|
| `src_train` | ~80% bản ghi nguồn (phân tầng theo miền × lớp) | học $G_y$ (có nhãn) |
| `src_val` | ~20% bản ghi nguồn | chọn epoch (quy tắc `src_val`), ngưỡng nguồn, ước lượng DEV |
| `tgt_train` (fold *k*) | bản ghi đích **ngoài** fold *k* — **không nhãn** | căn chỉnh miền (DANN, CDAN+E, CORAL, MMD, AdaBN) |
| `tgt_test` (fold *k*) | bản ghi đích **trong** fold *k* | **chỉ để báo cáo** |

- **Quy nạp**: mỗi bản ghi đích nằm trong `tgt_test` **đúng một lần** và **không bao giờ** có mặt (kể cả không nhãn)
  trong lượt huấn luyện sinh ra dự đoán của nó. `source_only` không dùng đích nên được huấn luyện **một lần mỗi seed**
  và dự đoán toàn bộ bản ghi đích (tương đương OOF, rẻ hơn K lần).
- `oracle` dùng **nhãn** của `tgt_train` (cùng fold) → cận trên tham chiếu, không phải UDA.
- **5 assertion cứng** được kiểm tra cho mọi phân hoạch: không giao nhau giữa 4 tập, phủ đích đúng một lần, đủ hai lớp.
''')

code(r'''
# =========================================================
# 6.1 Xác định bài toán chuyển miền / Define transfer tasks
# =========================================================
rec_counts = records.groupby(["domain", "label"]).size().unstack(fill_value=0).reindex(columns=[0, 1], fill_value=0)
HAS_BOTH = [d for d in DOMAINS_ALL if rec_counts.loc[d].min() >= 1]
ELIGIBLE_TGT = [d for d in DOMAINS_ALL if rec_counts.loc[d].min() >= cfg.min_records_per_class]

def _parse_task(spec):
    src, tgt = spec
    return [s.strip() for s in str(src).split("+") if s.strip()], str(tgt).strip()

if cfg.tasks:
    _raw = [_parse_task(t) for t in cfg.tasks]
else:
    _raw = [([d for d in HAS_BOTH if d != t], t) for t in ELIGIBLE_TGT]
TASKS = []
for srcs, tgt in _raw:
    bad = [d for d in srcs + [tgt] if d not in DOMAINS_ALL]
    assert not bad, f"Miền không tồn tại: {bad}. Các miền: {DOMAINS_ALL}"
    assert tgt not in srcs, f"Miền đích {tgt} không được nằm trong nguồn"
    if not srcs:
        continue
    src_lab = records[records.domain.isin(srcs)].label
    if src_lab.nunique() < 2 or rec_counts.loc[tgt].min() < cfg.min_records_per_class:
        print(f"  bỏ bài toán {'+'.join(srcs)} → {tgt}: thiếu bản ghi một lớp"); continue
    TASKS.append(dict(i=len(TASKS), src=srcs, tgt=tgt, name=f"{'+'.join(srcs)} → {tgt}",
                      slug=re.sub(r"[^A-Za-z0-9]+", "-", f"{'+'.join(srcs)}-to-{tgt}").strip("-")))
TASKS = TASKS[:cfg.max_tasks]
assert TASKS, "Không có bài toán chuyển miền hợp lệ (mỗi miền đích cần đủ bản ghi hai lớp)."

_tt = []
for T in TASKS:
    s = records[records.domain.isin(T["src"])]; t = records[records.domain == T["tgt"]]
    _tt.append(dict(task=T["name"], src_records=len(s), src_leak=int(s.label.sum()), tgt_records=len(t),
                    tgt_leak=int(t.label.sum()), src_frames=int(np.isin(dom_all, T["src"]).sum()),
                    tgt_frames=int((dom_all == T["tgt"]).sum()),
                    tgt_folds=int(min(cfg.tgt_folds, t.label.value_counts().min()))))
TASK_TAB = pd.DataFrame(_tt)
save_table(TASK_TAB, "transfer_tasks",
           "Transfer tasks (source → target), number of independent recordings (leak count) and frames per side, "
           "and number of recording-level folds on the target domain.", 0)
''')

code(r'''
# =========================================================
# 6.2 Phân hoạch quy nạp + 5 assertion cứng / Inductive partitions with hard checks
# =========================================================
FR_BY_UID = {k: np.asarray(v, dtype=np.int64) for k, v in frames.groupby("record_uid").indices.items()}
REC_LAB = records.set_index("record_uid").label

def frames_of(uids):
    uids = list(uids)
    return np.sort(np.concatenate([FR_BY_UID[u] for u in uids])) if uids else np.array([], dtype=np.int64)

def build_partition(T, seed):
    src = records[records.domain.isin(T["src"])].reset_index(drop=True)
    tgt = records[records.domain == T["tgt"]].reset_index(drop=True)
    n_val = max(2, int(round(cfg.src_val_frac * len(src))))
    strat = (src.domain + "|" + src.label.astype(str)).values
    if pd.Series(strat).value_counts().min() < 2 or n_val < len(set(strat)):
        strat = src.label.values                       # quá ít bản ghi để phân tầng theo miền × lớp
    a, b = next(StratifiedShuffleSplit(1, test_size=n_val, random_state=int(seed)).split(np.zeros(len(src)), strat))
    s_tr, s_va = src.record_uid.values[a], src.record_uid.values[b]
    t_uids = tgt.record_uid.values
    K = int(min(cfg.tgt_folds, tgt.label.value_counts().min()))
    if cfg.target_protocol == "transductive" or K < 2:
        mode, folds = "transductive", [(t_uids, t_uids)]
    else:
        mode = "inductive"
        skf = StratifiedKFold(K, shuffle=True, random_state=int(seed))
        folds = [(t_uids[i], t_uids[j]) for i, j in skf.split(np.zeros(len(tgt)), tgt.label.values)]
    P = dict(task=T["i"], seed=int(seed), mode=mode, src_tr=s_tr, src_va=s_va, tgt_all=t_uids, folds=folds,
             src_tr_idx=frames_of(s_tr), src_va_idx=frames_of(s_va), tgt_all_idx=frames_of(t_uids),
             fold_idx=[(frames_of(i), frames_of(j)) for i, j in folds])
    check_partition(P)
    return P

def check_partition(P):
    s_tr, s_va, t_all = set(P["src_tr"]), set(P["src_va"]), set(P["tgt_all"])
    assert not (s_tr & s_va), "LEAKAGE: bản ghi nằm ở cả src_train và src_val"
    assert not ((s_tr | s_va) & t_all), "LEAKAGE: bản ghi nằm ở cả nguồn và đích"
    assert REC_LAB[list(s_tr)].nunique() == 2 and REC_LAB[list(s_va)].nunique() == 2, "nguồn thiếu một lớp"
    seen = []
    for k, (tr, te) in enumerate(P["folds"]):
        if P["mode"] == "inductive":
            assert not (set(tr) & set(te)), f"LEAKAGE: fold {k} có bản ghi đích ở cả tgt_train và tgt_test"
        assert REC_LAB[list(te)].nunique() == 2, f"fold {k}: tgt_test thiếu một lớp"
        seen += list(te)
    if P["mode"] == "inductive":
        assert sorted(seen) == sorted(t_all), "mỗi bản ghi đích phải ở tgt_test ĐÚNG MỘT LẦN"
    return True

PARTS = {(T["i"], s): build_partition(T, s) for T in TASKS for s in cfg.seeds}
print(f"✓ {len(PARTS)} phân hoạch (bài toán × seed) qua 5 assertion; chế độ: "
      f"{sorted({P['mode'] for P in PARTS.values()})}")

_pr = []
for T in TASKS:
    P = PARTS[(T["i"], cfg.seeds[0])]
    for k, (tr, te) in enumerate(P["folds"]):
        _pr.append(dict(task=T["name"], fold=k, src_train_rec=len(P["src_tr"]), src_val_rec=len(P["src_va"]),
                        tgt_unlab_rec=len(tr), tgt_test_rec=len(te),
                        tgt_test_leak=int(REC_LAB[list(te)].sum()),
                        tgt_unlab_frames=len(P["fold_idx"][k][0]), tgt_test_frames=len(P["fold_idx"][k][1])))
save_table(pd.DataFrame(_pr), "partitions",
           f"Recording-level partitions for seed {cfg.seeds[0]} (inductive protocol: the unlabelled target data used "
           "for adaptation never contains the recordings being tested).", 0)

# ---- Hình: bản đồ phân hoạch của bài toán đầu tiên ----
T0, P0 = TASKS[0], PARTS[(0, cfg.seeds[0])]
cols = list(P0["src_tr"]) + list(P0["src_va"]) + list(P0["tgt_all"])
order = sorted(range(len(cols)), key=lambda i: (cols[i] not in set(P0["src_tr"]) | set(P0["src_va"]),
                                                REC_LAB[cols[i]], cols[i]))
cols = [cols[i] for i in order]; pos = {u: i for i, u in enumerate(cols)}
M = np.full((len(P0["folds"]), len(cols)), np.nan)
for k, (tr, te) in enumerate(P0["folds"]):
    for u in P0["src_tr"]: M[k, pos[u]] = 0
    for u in P0["src_va"]: M[k, pos[u]] = 1
    for u in tr: M[k, pos[u]] = 2
    for u in te: M[k, pos[u]] = 3
cmap = matplotlib.colors.ListedColormap(["#9FB7C9", "#2E6F95", "#F1C9B9", "#C8553D"])
fig, ax = plt.subplots(2, 1, figsize=(14, 1.2 + 0.45 * len(P0["folds"])), sharex=True,
                       gridspec_kw=dict(height_ratios=[max(2, len(P0["folds"])), .7], hspace=.12))
ax[0].imshow(M, aspect="auto", cmap=cmap, vmin=-.5, vmax=3.5, interpolation="nearest")
ax[0].set_yticks(range(len(P0["folds"]))); ax[0].set_yticklabels([f"fold {k}" for k in range(len(P0["folds"]))])
ax[0].grid(False)
ax[0].set_title(f"{T0['name']}: source-train | source-val | target unlabelled (adaptation) | target test")
ax[1].imshow(np.array([[REC_LAB[u] for u in cols]]), aspect="auto",
             cmap=matplotlib.colors.ListedColormap([PAL["noleak"], PAL["leak"]]), interpolation="nearest")
ax[1].set_yticks([0]); ax[1].set_yticklabels(["label"]); ax[1].set_xlabel("recording"); ax[1].grid(False)
savefig(fig, "partition_map",
        "Inductive evaluation protocol for the first transfer task: every target recording is tested exactly once, "
        "by a model whose unlabelled adaptation data excluded it (colours: light/dark blue = source train/validation, "
        "light/dark red = target unlabelled/test).")
''')

# =============================================================================
# 7. BỘ MÁY HUẤN LUYỆN
# =============================================================================
md(r'''
## 7. Bộ máy huấn luyện / Training engine

- **Một vòng lặp** = một lô nguồn (cân bằng lớp) + một lô đích không nhãn (+ một lô đích có nhãn với `oracle`/bán giám sát);
  hai miền đi **chung một lượt truyền** để BatchNorm thấy cả hai phân bố (chuẩn của DANN/CDAN).
- **Lịch** (Ganin et al., 2016): $\lambda_p = \lambda_{\max}\left(\frac{2}{1+e^{-10p}} - 1\right)$; LR: warm-up 5% + cosine (như v7);
  AdamW, AMP fp16, cắt gradient 5.0; nhãn làm mịn 0.05.
- **Đánh giá mỗi `eval_every` epoch** trên: `src_val` (BCE, AUC, ngưỡng Youden nguồn), `tgt_test` (lưu xác suất),
  và một mẫu con của `src_train`/`tgt_train` để tính **entropy đích**, **DEV** (You et al., 2019) và **PAD** của đặc trưng.
- Nhãn đích chỉ dùng cho cột *giám sát* (`tgt_auc_*`) và quy tắc `oracle` — **không bao giờ** cho quy tắc chính.
- Hai GPU: mỗi GPU chạy **một job độc lập** (không `DataParallel` → thống kê BatchNorm không bị chia đôi).
''')

code(r'''
# =========================================================
# 7.1 Dữ liệu thường trú GPU, bộ lấy mẫu, augmentation / resident data, samplers, augmentation
# =========================================================
DATA_GB = X_ALL.nbytes / 1024 ** 3
_RES, _RES_LOCK, _BUILD_LOCK = {}, threading.Lock(), threading.Lock()

def get_resident(device):
    if device.type != "cuda":
        return None
    key = str(device)
    with _RES_LOCK:
        if key not in _RES:
            if DATA_GB <= cfg.gpu_resident_max_gb:
                dt = torch.float32
            elif DATA_GB / 2 <= cfg.gpu_resident_max_gb:
                dt = torch.float16
            else:
                _RES[key] = None
                print(f"  [{device}] dữ liệu {DATA_GB:.1f} GB > ngưỡng -> đọc từ RAM mỗi lô")
                return None
            R = torch.empty(X_ALL.shape, dtype=dt, device=device)
            for s in range(0, len(X_ALL), 4096):
                R[s:s + 4096] = torch.from_numpy(np.asarray(X_ALL[s:s + 4096])).to(device, dtype=dt)
            _RES[key] = R
            print(f"  [{device}] nạp {DATA_GB * (1 if dt == torch.float32 else .5):.2f} GB frame ({dt})")
    return _RES[key]

def wave_augment(xb, g, hp):
    # [V2] ngẫu nhiên hoá chuỗi thu | đáp ứng kênh FIR ngẫu nhiên | nhiễu Gauss theo SNR | dịch vòng
    #      — theo lô, trên thiết bị của lô, mỗi mẫu một tham số riêng.
    B, T = xb.shape; dev = xb.device
    rnd = lambda n: torch.rand(n, device=dev, generator=g)
    m = rnd(B) < hp.get("chain_p", 0.0)
    if bool(m.any()):
        # Thông cao Butterworth bậc 2 (fc ngẫu nhiên: mô phỏng cảm biến/máy ghi cắt tần thấp khác nhau — đúng
        # loại lệch giữa chuỗi R và S ở v1), độ dốc phổ ±dB/octave, thông thấp bậc 4; pha 0, miền tần số.
        X = torch.fft.rfft(xb, dim=1)
        f = torch.fft.rfftfreq(T, d=1.0 / cfg.target_sr).to(dev)
        lo, hi = cfg.aug_hp_hz; fc = (lo + rnd(B) * (hi - lo))[:, None]
        t0, t1 = cfg.aug_tilt_db_oct; sl = (t0 + rnd(B) * (t1 - t0))[:, None]
        l0, l1 = cfg.aug_lp_hz; fl = (l0 + rnd(B) * (l1 - l0))[:, None]
        H = (1.0 / torch.sqrt(1.0 + (fc / f.clamp_min(1.0)) ** 4)
             * (f.clamp_min(20.0) / 1000.0) ** (sl / 6.0206)
             * 1.0 / torch.sqrt(1.0 + (f / fl) ** 8))
        H = torch.where(m[:, None], H, torch.ones_like(H))
        xb = torch.fft.irfft(X * H, n=T, dim=1)
    m = rnd(B) < hp["fir_p"]
    if bool(m.any()):
        taps = cfg.aug_fir_taps + (1 - cfg.aug_fir_taps % 2)
        h = torch.randn(B, 1, taps, device=dev, generator=g) * torch.hann_window(taps, periodic=False, device=dev)
        h = h / h.norm(dim=2, keepdim=True).clamp_min(1e-12)
        delta = torch.zeros(B, 1, taps, device=dev); delta[:, 0, taps // 2] = 1.0
        h = torch.where(m.view(B, 1, 1), h, delta)
        xb = F.conv1d(xb.view(1, B, T), h, padding=taps // 2, groups=B).view(B, T)
    m = (rnd(B) < hp["noise_p"]).float()
    lo, hi = cfg.aug_noise_snr_db
    snr = rnd(B) * (hi - lo) + lo
    sigma = torch.sqrt(((xb ** 2).mean(1) + 1e-12) / (10 ** (snr / 10.0))) * m
    xb = xb + torch.randn(xb.shape, device=dev, generator=g) * sigma[:, None]
    m = rnd(B) < hp["shift_p"]
    if bool(m.any()):
        sh = torch.randint(0, T, (B,), device=dev, generator=g) * m
        xb = torch.gather(xb, 1, (torch.arange(T, device=dev)[None] - sh[:, None]) % T)
    return torch.nan_to_num(xb)

def fetch(idx, device, augment=False, g=None, hp=None):
    idx = np.asarray(idx, dtype=np.int64)
    R = get_resident(device)
    if R is not None:
        xb = R.index_select(0, torch.as_tensor(idx, device=device)).float()
    else:
        xb = torch.from_numpy(np.asarray(X_ALL[idx], dtype=np.float32)).to(device, non_blocking=True)
    return wave_augment(xb, g, hp) if augment else xb

class Sampler:
    # Lấy mẫu "vô hạn" theo hoán vị (phủ đều dữ liệu); balanced=True -> nửa leak nửa noleak.
    def __init__(self, idx, bs, seed, balanced=False):
        idx = np.asarray(idx, dtype=np.int64)
        self.rng = np.random.default_rng(int(seed)); self.bs = int(bs)
        pools = [idx[y_all[idx] == c] for c in (0, 1)] if balanced else [idx]
        self.pools = [p for p in pools if len(p)]
        assert self.pools, "Sampler rỗng"
        self.perm = [self.rng.permutation(p) for p in self.pools]; self.ptr = [0] * len(self.pools)
    def _take(self, k, n):
        out = []
        while n > 0:
            if self.ptr[k] >= len(self.perm[k]):
                self.perm[k] = self.rng.permutation(self.pools[k]); self.ptr[k] = 0
            m = min(n, len(self.perm[k]) - self.ptr[k])
            out.append(self.perm[k][self.ptr[k]:self.ptr[k] + m]); self.ptr[k] += m; n -= m
        return np.concatenate(out)
    def next(self):
        if len(self.pools) == 1:
            return self._take(0, self.bs)
        h = self.bs // 2
        return np.concatenate([self._take(0, self.bs - h), self._take(1, h)])

class BagSampler:
    # [V2-RCST] lô đích = n_rec bản ghi × k frame mỗi bản ghi -> phạt được sự KHÔNG nhất quán trong bản ghi.
    def __init__(self, idx, n_rec, k, seed):
        idx = np.asarray(idx, dtype=np.int64); u = uid_all[idx]
        o = np.argsort(u, kind="stable"); us = u[o]
        self.groups = np.split(idx[o], np.flatnonzero(us[1:] != us[:-1]) + 1)
        self.rng = np.random.default_rng(int(seed)); self.n_rec, self.k = int(n_rec), int(k)
    def next(self):
        r = self.rng.choice(len(self.groups), self.n_rec, replace=len(self.groups) < self.n_rec)
        out = [self.rng.choice(self.groups[i], self.k, replace=len(self.groups[i]) < self.k) for i in r]
        return np.concatenate(out), np.repeat(np.arange(self.n_rec), self.k)

def subsample(idx, n, seed):
    idx = np.asarray(idx)
    return idx if len(idx) <= n else np.sort(np.random.default_rng(seed).choice(idx, n, replace=False))

print(f"Dữ liệu frame: {DATA_GB:.2f} GB | thường trú GPU nếu <= {cfg.gpu_resident_max_gb} GB "
      f"(fp16 nếu <= {2 * cfg.gpu_resident_max_gb} GB)")
''')

code(r'''
# =========================================================
# 7.2 Lịch λ_p và LR + Hình / lambda and learning-rate schedules
# =========================================================
def grl_lambda(it, lam_max):
    p = it / max(1, TOTAL_ITERS - 1)
    return float(lam_max * (2.0 / (1.0 + math.exp(-cfg.lambda_gamma * p)) - 1.0))

def lr_factor(step):
    warm = max(1, int(cfg.warmup_frac * TOTAL_ITERS))
    if step < warm:
        return float(step + 1) / warm
    prog = min(1.0, (step - warm) / max(1, TOTAL_ITERS - warm))
    return 1e-3 + (1 - 1e-3) * 0.5 * (1 + math.cos(math.pi * prog))

_its = np.arange(TOTAL_ITERS)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.2))
for lm in sorted(set((cfg.lambda_max,) + tuple(cfg.lambda_grid))):
    ax[0].plot(_its / cfg.iters_per_epoch, [grl_lambda(i, lm) for i in _its],
               lw=2.2 if lm == cfg.lambda_max else 1, label=f"λ_max = {lm:g}" + (" (main)" if lm == cfg.lambda_max else ""))
ax[0].axvline(SELECT_FROM_EPOCH, color="k", ls=":", lw=1)
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("λ_p"); ax[0].set_title("(a) GRL coefficient schedule"); ax[0].legend()
ax[1].plot(_its / cfg.iters_per_epoch, [cfg.lr * lr_factor(i) for i in _its], color=PAL["ok"])
ax[1].axvline(SELECT_FROM_EPOCH, color="k", ls=":", lw=1, label="earliest selectable epoch")
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("learning rate"); ax[1].set_title("(b) Learning-rate schedule"); ax[1].legend()
savefig(fig, "schedules",
        f"Training schedules: (a) gradient-reversal coefficient λ_p = λ_max(2/(1+exp(−{cfg.lambda_gamma:g}p)) − 1) and "
        "(b) warm-up + cosine learning rate. Checkpoints are never selected before the dotted line.")
''')

code(r'''
# =========================================================
# 7.3 Suy luận, chỉ số, DEV và PAD / inference, metrics, DEV and PAD
# =========================================================
def amp_on(device):
    return bool(cfg.amp and device.type == "cuda")

@torch.no_grad()
def infer(model, idx, device, want_feat=False, bs=256):
    model.eval(); P, Fs = [], []
    idx = np.asarray(idx)
    for s in range(0, len(idx), bs):
        xb = fetch(idx[s:s + bs], device)
        with amp_autocast(device, amp_on(device)):
            logit, f = model(xb, return_feat=True)
        P.append(torch.sigmoid(logit.float()).clamp(0, 1).cpu().numpy())
        if want_feat:
            Fs.append(f.float().cpu().numpy())
    p = np.nan_to_num(np.concatenate(P) if P else np.array([]), nan=0.5)
    return (p, np.concatenate(Fs)) if want_feat else p

def frame_bce(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))

def rec_probs_from(idx, p):
    g = pd.DataFrame(dict(uid=uid_all[np.asarray(idx)], y=y_all[np.asarray(idx)], p=p))
    return g.groupby("uid").agg(y=("y", "first"), p=("p", "mean"))

def safe_auc(y, p):
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan

def youden(y, p):
    if len(np.unique(y)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y, p)
    thr = np.where(np.isfinite(thr), thr, 1.0)
    return float(np.clip(thr[int(np.argmax(tpr - fpr))], 0, 1))

def _lr_domain(Xs, Xt):
    X = np.vstack([Xs, Xt]); y = np.r_[np.zeros(len(Xs)), np.ones(len(Xt))]
    sc = StandardScaler().fit(X)
    return sc, LogisticRegression(C=1.0, max_iter=500).fit(sc.transform(X), y)

def pad_score(Xs, Xt, seed=0, k=2):
    # Proxy A-distance = 2(1 - 2·err) của bộ phân lớp miền tuyến tính, CV k-fold trên mẫu cân bằng.
    n = min(len(Xs), len(Xt))
    if n < 10:
        return np.nan
    rng = np.random.default_rng(seed)
    X = np.vstack([Xs[rng.choice(len(Xs), n, replace=False)], Xt[rng.choice(len(Xt), n, replace=False)]])
    y = np.r_[np.zeros(n), np.ones(n)]; errs = []
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=1.0, max_iter=500).fit(sc.transform(X[tr]), y[tr])
        errs.append(np.mean(clf.predict(sc.transform(X[te])) != y[te]))
    return pad_from_err(float(np.mean(errs)))

def dev_risk(f_src_tr, f_src_va, loss_va, f_tgt):
    # Deep Embedded Validation (You et al., 2019): rủi ro nguồn-val có trọng số mật độ đích/nguồn,
    # cộng biến điều khiển để giảm phương sai. KHÔNG dùng nhãn đích. Thấp hơn = tốt hơn.
    try:
        sc, clf = _lr_domain(f_src_tr, f_tgt)
        q = clf.predict_proba(sc.transform(f_src_va))[:, 1].clip(1e-4, 1 - 1e-4)
        w = np.clip(q / (1 - q) * (len(f_src_tr) / len(f_tgt)), 0, 20)
        L = w * loss_va
        C = np.cov(np.vstack([L, w]))
        eta = -C[0, 1] / max(C[1, 1], 1e-12)
        return float(L.mean() + eta * w.mean() - eta)
    except Exception:
        return np.nan

def _snapshot(model):
    return {k: (v.detach().half() if v.is_floating_point() else v.detach()).cpu().clone()
            for k, v in model.state_dict().items()}

def load_state(model, state):
    model.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in state.items()})
    return model

RULES = ("last", "src_val", "dev", "oracle")
print("Inference/metrics/DEV/PAD OK | quy tắc chọn:", RULES, "| chính:", cfg.select_rule)
''')

code(r'''
# =========================================================
# 7.4 Huấn luyện MỘT job (bài toán × seed × fold × phương pháp) / single-job training engine
# =========================================================
# [V2.1] Trạng thái từng GPU (job, epoch, vòng lặp, thời điểm tiến triển cuối) do train_job cập nhật -> luồng chính
#        in nhịp tim và phát hiện treo; _STOP cho phép dừng sạch các luồng khi người dùng ngắt ô (Interrupt).
_PROG, _STOP = {}, threading.Event()

class JobCancelled(RuntimeError):
    pass

def job_key(j):
    return f"{j['variant']}|t{j['task']}|s{j['seed']}|f{j['fold']}|{j['method']}|sh{j.get('shots', 0)}"

def _seed_job(seed, device):
    # [V2.1] chỉ gieo RNG của CPU và của CHÍNH GPU này. set_seed() gọi manual_seed_all -> đặt lại RNG của GPU kia
    #        giữa lúc job khác đang chạy ở đó (dropout phụ thuộc thời điểm, kết quả không tái lập được).
    random.seed(seed); np.random.seed(seed); torch.default_generator.manual_seed(seed)
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.manual_seed(seed)

def method_hp(job):
    hp = dict(lambda_max=cfg.lambda_max, fir_p=cfg.aug_fir_p, noise_p=cfg.aug_noise_p,
              shift_p=cfg.aug_shift_p, chain_p=cfg.aug_chain_p, augment=cfg.augment,
              balanced=cfg.class_balanced_source, disc_hidden=cfg.disc_hidden, disc_dropout=cfg.disc_dropout,
              disc_lr_mult=cfg.disc_lr_mult, rcst_pl=True, rcst_cons=True)
    hp.update(job.get("hp", {}))
    if not hp["augment"]:
        hp["fir_p"] = hp["noise_p"] = hp["shift_p"] = hp["chain_p"] = 0.0
    return hp

def job_target_split(job):
    P = PARTS[(job["task"], job["seed"])]
    if job["fold"] is None:
        return P["tgt_all_idx"], P["tgt_all_idx"]
    return P["fold_idx"][job["fold"]]

def labelled_target(job, tgt_tr):
    shots = job.get("shots", -1 if job["method"] in ("oracle", "target_only") else 0)
    if shots == 0:
        return np.array([], dtype=np.int64)
    uids = pd.unique(uid_all[np.asarray(tgt_tr)])
    if shots > 0:
        rng = np.random.default_rng(1000 * job["seed"] + (job["fold"] or 0))
        pick = []
        for c in (0, 1):
            u = np.array([x for x in uids if REC_LAB[x] == c])
            pick += list(rng.choice(u, min(shots, len(u)), replace=False))
        uids = pick
    return frames_of(uids)

def rcst_pseudo_labels(model, tgt_tr, device, frac):
    # [V2-RCST] Nhãn giả MỨC BẢN GHI: xác suất bản ghi = trung bình xác suất frame (chế độ eval, không augment);
    # chọn cân bằng theo THỨ HẠNG (frac/2 bản ghi cao nhất -> leak, frac/2 thấp nhất -> noleak) nên không phụ
    # thuộc ngưỡng 0.5 (vốn lệch khi có dịch chuyển miền). Nhãn thật chỉ dùng để GIÁM SÁT chất lượng nhãn giả.
    tgt_tr = np.asarray(tgt_tr)
    p = infer(model, tgt_tr, device)
    g = rec_probs_from(tgt_tr, p).sort_values("p")
    n_each = max(1, int(round(frac * len(g) / 2)))
    neg, pos = list(g.index[:n_each]), [u for u in g.index[::-1][:n_each] if u not in set(g.index[:n_each])]
    vec = np.full(N_FRAMES, -1.0, dtype=np.float32)
    for u in pos:
        vec[FR_BY_UID[u]] = 1.0
    for u in neg:
        vec[FR_BY_UID[u]] = 0.0
    lab = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    acc_rec = float(np.mean(lab == g.loc[pos + neg, "y"].values))
    o = np.argsort(p); k = int(round(frac * len(p) / 2))            # đối chứng: nhãn giả MỨC FRAME cùng độ phủ
    yt = y_all[tgt_tr]
    acc_fr = float(np.mean(np.r_[yt[o[:k]] == 0, yt[o[::-1][:k]] == 1])) if k else np.nan
    return torch.as_tensor(vec, device=device), dict(pl_cov=(len(pos) + len(neg)) / len(g),
                                                     pl_acc_rec=acc_rec, pl_acc_frame=acc_fr)

def train_job(job, device, verbose=False):
    t0 = time.time()
    P = PARTS[(job["task"], job["seed"])]
    method, hp = job["method"], method_hp(job)
    tgt_tr, tgt_te = job_target_split(job)
    lab_idx = labelled_target(job, tgt_tr)
    use_tgt = method in ADAPT_METHODS
    src_tr, src_va = P["src_tr_idx"], P["src_va_idx"]
    seed = 100 * job["seed"] + (0 if job["fold"] is None else job["fold"] + 1) + 7 * job.get("seed_offset", 0)

    pr = _PROG.get(str(device))                    # [V2.1] tiến độ cho nhịp tim/phát hiện treo (None khi chạy lẻ)
    with _BUILD_LOCK:
        _seed_job(seed, device)
        model = LeakNet().to(device)
        disc = None
        if method in ADV_METHODS:
            disc = DomainDisc(cfg.bottleneck_dim * (2 if method.startswith("cdan") else 1),
                              hidden=hp["disc_hidden"], p=hp["disc_dropout"]).to(device)
    if hasattr(model, "specaug"):
        model.specaug.generator = torch.Generator().manual_seed(seed * 7919 + 13)
    g = torch.Generator(device=device); g.manual_seed(seed + 1)
    Y = torch.as_tensor(y_all, device=device)
    if method == "target_only":                   # cận trên thứ hai: CHỈ đích có nhãn (cùng fold), không nguồn
        s_samp, l_samp = Sampler(lab_idx, cfg.batch_size, seed + 2, balanced=True), None
    else:
        s_samp = Sampler(src_tr, cfg.batch_size, seed + 2, balanced=hp["balanced"])
        l_samp = Sampler(lab_idx, max(8, cfg.batch_size // 2), seed + 4, balanced=True) if len(lab_idx) else None
    bag = method == "rcst"
    t_samp = None
    if use_tgt:
        t_samp = (BagSampler(tgt_tr, cfg.rcst_bag[0], cfg.rcst_bag[1], seed + 3) if bag
                  else Sampler(tgt_tr, cfg.batch_size, seed + 3))
    pw = None
    if not hp["balanced"] and method != "target_only":
        pos = float(y_all[src_tr].sum()); pw = torch.tensor([(len(src_tr) - pos) / max(pos, 1.0)], device=device)

    groups = [{"params": model.parameters(), "lr": cfg.lr}]
    if disc is not None:
        groups.append({"params": disc.parameters(), "lr": cfg.lr * hp["disc_lr_mult"]})
    opt = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    scaler = make_grad_scaler(amp_on(device))
    all_params = list(model.parameters()) + (list(disc.parameters()) if disc is not None else [])
    eps = cfg.label_smoothing
    rc_start = int(math.ceil(cfg.rcst_start_frac * cfg.epochs))
    pl_vec, pl_info = None, {}

    ep_rows, hist_rows, preds, snaps = [], [], {}, {}
    it, n_skip = 0, 0
    for ep in range(cfg.epochs):
        if pr is not None:
            pr.update(ep=ep + 1, t=time.time())
        if bag and hp["rcst_pl"] and ep >= rc_start:      # làm mới nhãn giả mức bản ghi đầu mỗi epoch
            s0, s1 = cfg.rcst_sel
            frac = s0 + (s1 - s0) * (ep - rc_start) / max(1, cfg.epochs - 1 - rc_start)
            pl_vec, pl_info = rcst_pseudo_labels(model, tgt_tr, device, frac)
        w_pl = (cfg.rcst_pl_weight * min(1.0, (ep - rc_start + 1) / max(1.0, (cfg.epochs - rc_start) / 2))
                if pl_vec is not None else 0.0)
        model.train()
        if disc is not None:
            disc.train()
        acc = defaultdict(float); nb = 0
        for _ in range(cfg.iters_per_epoch):
            if _STOP.is_set():
                raise JobCancelled(job_key(job))
            if pr is not None:
                pr.update(it=it, t=time.time())
            lam = grl_lambda(it, hp["lambda_max"])
            i_s = s_samp.next()
            parts = [fetch(i_s, device, hp["augment"], g, hp)]
            ns, nt = len(i_s), 0
            if use_tgt:
                if bag:
                    i_t, grp = t_samp.next(); grp = torch.as_tensor(grp, device=device)
                else:
                    i_t = t_samp.next()
                parts.append(fetch(i_t, device, hp["augment"], g, hp)); nt = len(i_t)
            y_lab = Y[torch.as_tensor(i_s, device=device)]
            if l_samp is not None:
                i_l = l_samp.next(); parts.append(fetch(i_l, device, hp["augment"], g, hp))
                y_lab = torch.cat([y_lab, Y[torch.as_tensor(i_l, device=device)]])
            x = torch.cat(parts)
            with amp_autocast(device, amp_on(device)):
                logit, f = model(x, return_feat=True)
                logit_lab = torch.cat([logit[:ns], logit[ns + nt:]])
                loss_cls = F.binary_cross_entropy_with_logits(logit_lab.float(), y_lab * (1 - eps) + eps / 2,
                                                              pos_weight=pw)
                loss_da = torch.zeros((), device=device); loss_st = torch.zeros((), device=device)
                d_acc = np.nan
                if method in ADV_METHODS:
                    dy = torch.cat([torch.ones(ns, device=device), torch.zeros(nt, device=device)])
                    h = cdan_input(f[:ns + nt], logit[:ns + nt]) if method.startswith("cdan") else f[:ns + nt]
                    d = disc(h, lam).float()
                    if method == "cdan_e":
                        w = entropy_weights(logit[:ns], logit[ns:ns + nt])
                        loss_da = (w * F.binary_cross_entropy_with_logits(d, dy, reduction="none")).sum() / w.sum()
                    else:
                        loss_da = F.binary_cross_entropy_with_logits(d, dy)
                    d_acc = float(((d > 0).float() == dy).float().mean())
                elif method == "coral":
                    loss_da = grl_lambda(it, cfg.coral_weight) * coral_loss(f[:ns], f[ns:ns + nt])
                elif method == "mmd":
                    loss_da = grl_lambda(it, cfg.mmd_weight) * mmd_loss(f[:ns], f[ns:ns + nt])
                elif method == "mcc":
                    loss_da = grl_lambda(it, cfg.mcc_weight) * mcc_loss(logit[ns:ns + nt])
                # joint_bn: KHÔNG có hàm căn chỉnh — chỉ lô trộn nguồn + đích đi qua BatchNorm
                if bag:
                    lt = logit[ns:ns + nt]
                    if hp["rcst_cons"]:
                        loss_st = loss_st + grl_lambda(it, cfg.rcst_cons_weight) * record_consistency(
                            lt, grp, cfg.rcst_bag[0])
                    if pl_vec is not None:
                        yp = pl_vec[torch.as_tensor(i_t, device=device)]
                        mk = yp >= 0
                        if bool(mk.any()):
                            loss_st = loss_st + w_pl * F.binary_cross_entropy_with_logits(
                                lt[mk].float(), yp[mk] * (1 - eps) + eps / 2)
                loss = loss_cls + loss_da + loss_st
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                n_skip += 1; it += 1; continue
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(all_params, cfg.grad_clip)
            scaler.step(opt); scaler.update(); sched.step()
            acc["loss_cls"] += float(loss_cls); acc["loss_da"] += float(loss_da); acc["loss_st"] += float(loss_st)
            if np.isfinite(d_acc):
                acc["disc_acc"] += d_acc
            nb += 1; it += 1
        ep_rows.append(dict(epoch=ep + 1, it=it, lam=grl_lambda(max(0, it - 1), hp["lambda_max"]),
                            lr=opt.param_groups[0]["lr"], loss_cls=acc["loss_cls"] / max(nb, 1),
                            loss_da=acc["loss_da"] / max(nb, 1), loss_st=acc["loss_st"] / max(nb, 1),
                            disc_acc=(acc["disc_acc"] / max(nb, 1)) if method in ADV_METHODS else np.nan,
                            **pl_info))

        if (ep + 1) % cfg.eval_every == 0 or ep == cfg.epochs - 1:
            p_va, f_va = infer(model, src_va, device, want_feat=True)
            l_va = frame_bce(y_all[src_va], p_va)
            g_va = rec_probs_from(src_va, p_va)
            p_te = infer(model, tgt_te, device)
            i_tu = subsample(tgt_tr, cfg.dev_max, seed + ep)
            p_tu, f_tu = infer(model, i_tu, device, want_feat=True)
            _, f_su = infer(model, subsample(src_tr, cfg.dev_max, seed + ep + 1), device, want_feat=True)
            g_te = rec_probs_from(tgt_te, p_te)
            d_eval = np.nan
            if disc is not None:
                # [V2] độ chính xác bộ phân biệt ở CHẾ ĐỘ EVAL (không dropout/augment), cân bằng hai miền:
                # v1 chỉ đo trên lô huấn luyện nhiễu nên "≈ 0.5" không chứng minh được bất biến miền.
                disc.eval()
                with torch.no_grad():
                    n_ = min(len(f_va), len(f_tu))
                    hs = torch.as_tensor(f_va[:n_], device=device); ht = torch.as_tensor(f_tu[:n_], device=device)
                    if method.startswith("cdan"):
                        lg = lambda q: torch.logit(torch.as_tensor(q[:n_], device=device).float(), eps=1e-6)
                        hs, ht = cdan_input(hs, lg(p_va)), cdan_input(ht, lg(p_tu))
                    d_eval = 0.5 * float((disc(hs).float() > 0).float().mean() + (disc(ht).float() <= 0).float().mean())
            hist_rows.append(dict(
                epoch=ep + 1, src_val_bce=float(l_va.mean()), src_val_auc=safe_auc(y_all[src_va], p_va),
                src_val_rec_auc=safe_auc(g_va.y, g_va.p), thr_src=youden(g_va.y.values, g_va.p.values),
                tgt_entropy=float(np.mean(frame_bce(p_tu, p_tu))) if len(p_tu) else np.nan,
                dev=dev_risk(f_su, f_va, l_va, f_tu), pad=pad_score(f_su, f_tu, seed=ep), disc_acc_eval=d_eval,
                # --- chỉ để GIÁM SÁT / quy tắc oracle (dùng nhãn đích) ---
                tgt_auc_frame=safe_auc(y_all[tgt_te], p_te), tgt_auc_rec=safe_auc(g_te.y, g_te.p)))
            preds[ep + 1] = p_te.astype(np.float16)
            if job.get("keep_state"):
                snaps[ep + 1] = _snapshot(model)
            if verbose:
                h = hist_rows[-1]
                print(f"   ep{ep + 1:03d} cls {ep_rows[-1]['loss_cls']:.3f} da {ep_rows[-1]['loss_da']:.3f} "
                      f"st {ep_rows[-1]['loss_st']:.3f} dacc {ep_rows[-1]['disc_acc']:.2f}/eval {d_eval:.2f} | "
                      f"srcval auc {h['src_val_auc']:.3f} | PAD {h['pad']:.2f} | [giám sát] tgt rec-AUC "
                      f"{h['tgt_auc_rec']:.3f}" + (f" | PL acc bản ghi {pl_info['pl_acc_rec']:.2f} / frame "
                                                    f"{pl_info['pl_acc_frame']:.2f}" if pl_info else ""))

    H = pd.DataFrame(hist_rows)
    el = H[H.epoch >= SELECT_FROM_EPOCH]
    el = el if len(el) else H
    last = int(el.epoch.iloc[-1])
    def _arg(col, fn):
        v = el[col]
        return int(el.loc[getattr(v, fn)(), "epoch"]) if v.notna().any() else last
    sel = dict(last=last, src_val=_arg("src_val_bce", "idxmin"), dev=_arg("dev", "idxmin"),
               oracle=_arg("tgt_auc_rec", "idxmax"))
    Hs = H.set_index("epoch")
    res = dict(key=job_key(job), variant=job["variant"], task=job["task"], seed=job["seed"], fold=job["fold"],
               method=method, shots=job.get("shots", 0), hp=hp, hist=H, ep_hist=pd.DataFrame(ep_rows),
               te_idx=np.asarray(tgt_te), sel=sel,
               pred={r: preds[e].astype(np.float32) for r, e in sel.items()},
               thr={r: float(Hs.loc[e, "thr_src"]) for r, e in sel.items()},
               src_val_rec_auc={r: float(Hs.loc[e, "src_val_rec_auc"]) for r, e in sel.items()},
               state=snaps.get(sel[cfg.select_rule]) if job.get("keep_state") else None,
               n_skip=int(n_skip), time_s=time.time() - t0, device=str(device))
    del model, disc, snaps; gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return res

# ---- Thử nhanh một job để bắt lỗi sớm (2 epoch ngắn, không lưu): RCST đi qua gần hết các nhánh mã ----
_save = (cfg.epochs, cfg.iters_per_epoch, cfg.eval_every)
cfg.epochs, cfg.iters_per_epoch, cfg.eval_every = 2, 2, 1
TOTAL_ITERS = cfg.epochs * cfg.iters_per_epoch; _sfe = SELECT_FROM_EPOCH; SELECT_FROM_EPOCH = 1
_probe = train_job(dict(variant="probe", task=0, seed=cfg.seeds[0], fold=0, method="rcst"), DEVICE, verbose=True)
cfg.epochs, cfg.iters_per_epoch, cfg.eval_every = _save
TOTAL_ITERS = cfg.epochs * cfg.iters_per_epoch; SELECT_FROM_EPOCH = _sfe
print(f"✓ job thử RCST chạy trọn ({_probe['time_s']:.1f}s, {len(_probe['te_idx'])} frame kiểm thử).")
''')

code(r'''
# =========================================================
# 7.5 Hàng đợi job trên các GPU + AdaBN + lưu/khôi phục / multi-GPU job queue, AdaBN, resume
# =========================================================
RES_PATH = OUT / "ada_results.pkl"
RUN_LOG = OUT / "run_log.txt"
# [V2.1] Dấu vân tay chỉ gồm những gì làm THAY ĐỔI kết quả của từng job (dữ liệu, front-end, siêu tham số). Phạm vi
#        chạy (seed, phương pháp, biến thể ablation...) đã nằm trong khoá job nên không vào dấu vân tay -> có thể
#        chạy theo giai đoạn (seeds=(42,) trước, sau đó (42, 43, 44)) mà vẫn dùng lại job cũ.
_SCOPE_KEYS = {"seeds", "methods", "ablation_variants", "ablation_seeds", "sweep_seeds", "lambda_grid", "run_ssda",
               "ssda_shots", "run_lambda_sweep", "run_ablation", "run_band_occlusion", "run_shortcut_check",
               "resume", "resume_dirs", "parallel_jobs", "max_parallel", "save_models", "out_dir", "fig_dpi",
               "heartbeat_min", "stall_min", "save_every_min", "bootstrap_n", "tsne_max_points"}
RUN_FP = hashlib.sha1((DATA_FP + json.dumps({k: v for k, v in _cfg_print.items() if k not in _SCOPE_KEYS},
                                            sort_keys=True, default=str)
                       + f"{F_MIN_EFF}|{F_MAX_EFF}").encode()).hexdigest()[:12]

_LOG_LOCK = threading.Lock()
def log(msg, echo=True):
    # In ra ô (flush ngay) VÀ ghi vào run_log.txt có dấu thời gian: nếu giao diện Kaggle ngừng nhận output
    # (mất kết nối trình duyệt), tiến độ vẫn còn trong tệp và trong Log của phiên bản "Save & Run All".
    if echo:
        print(msg, flush=True)
    with _LOG_LOCK:
        with open(RUN_LOG, "a", encoding="utf-8") as fh:
            fh.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")

def _result_files():
    out = [RES_PATH]
    for d in cfg.resume_dirs:
        d = Path(d)
        if d.exists():
            for pat in ("*/ada_results.pkl", "*/*/ada_results.pkl", "*/*/*/ada_results.pkl", "*/*/*/*/ada_results.pkl"):
                out += sorted(d.glob(pat))
    return [p for p in dict.fromkeys(out) if p.exists()]

RESULTS = {}
if cfg.resume:
    for _p in _result_files():
        try:
            _blob = pickle.load(open(_p, "rb"))
        except Exception as e:
            print(f"Không đọc được {_p}: {type(e).__name__}"); continue
        if _blob.get("fp") != RUN_FP:
            print(f"Bỏ qua {_p}: thuộc cấu hình khác ({_blob.get('fp')} ≠ {RUN_FP})."); continue
        _new = {k: v for k, v in _blob["results"].items() if k not in RESULTS}
        RESULTS.update(_new)
        print(f"Khôi phục {len(_new)} job từ {_p} (cấu hình {RUN_FP}).")
print(f"Dấu vân tay cấu hình: {RUN_FP} | {len(RESULTS)} job có sẵn | nhật ký: {RUN_LOG}")

_SAVE_LOCK = threading.Lock()
def save_results():
    with _SAVE_LOCK:
        tmp = RES_PATH.with_suffix(".tmp")
        pickle.dump(dict(fp=RUN_FP, results=RESULTS), open(tmp, "wb"))
        os.replace(tmp, RES_PATH)

def _job_devices():
    ngpu = torch.cuda.device_count() if DEVICE.type == "cuda" else 0
    return ([torch.device(f"cuda:{i}") for i in range(min(ngpu, cfg.max_parallel))]
            if (cfg.parallel_jobs and ngpu > 1) else [DEVICE])

def _job_name(j):
    return f"{j['variant']} {TASKS[j['task']]['name']} s{j['seed']} f{j['fold']} {j['method']}"

def _dump_stacks(why):
    frames = sys._current_frames()
    lines = [f"!! {why} — stack trace của mọi luồng:"]
    for t in threading.enumerate():
        lines.append(f"--- luồng {t.name} ---")
        lines += [l.rstrip() for l in traceback.format_stack(frames.get(t.ident))] if t.ident in frames else []
    log("\n".join(lines))

def run_jobs(jobs, title=""):
    jobs = [j for j in jobs if job_key(j) not in RESULTS]
    if not jobs:
        log(f"[{title}] mọi job đã có kết quả."); return
    devices = _job_devices()
    log(f"[{title}] {len(jobs)} job trên {[str(d) for d in devices]} | nhịp tim mỗi {cfg.heartbeat_min:g} phút "
        f"| nhật ký {RUN_LOG.name}")
    q = queue.Queue()
    for j in jobs:
        q.put(j)
    lock, done, t0, failed = threading.Lock(), [0], time.time(), []
    last_save = [time.time()]
    _STOP.clear(); _PROG.clear()

    def worker(dev):
        while not _STOP.is_set():
            try:
                j = q.get_nowait()
            except queue.Empty:
                return
            _PROG[str(dev)] = dict(name=_job_name(j), ep=0, it=0, t=time.time(), t_job=time.time())
            log(f"  ▶ {_job_name(j)} @{dev}", echo=False)
            try:
                r = train_job(j, dev)
                with lock:
                    RESULTS[r["key"]] = r; done[0] += 1
                    el = time.time() - t0; eta = el / done[0] * (len(jobs) - done[0])
                    h = r["hist"].set_index("epoch").loc[r["sel"][cfg.select_rule]]
                    log(f"  [{done[0]:3d}/{len(jobs)}] {j['variant']:>10s} {TASKS[j['task']]['name']:>18s} "
                        f"s{j['seed']} f{j['fold']} {j['method']:<11s} {r['time_s']:5.0f}s @{dev} | "
                        f"srcval AUC {h.src_val_auc:.3f} | [giám sát] tgt rec-AUC {h.tgt_auc_rec:.3f} | ETA {eta / 60:.0f} ph")
                    if time.time() - last_save[0] >= 60 * cfg.save_every_min:
                        save_results(); last_save[0] = time.time()
            except JobCancelled:
                return
            except Exception as e:
                with lock:
                    failed.append(j)
                    log(f"  !! {job_key(j)} lỗi trên {dev}: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            finally:
                _PROG.pop(str(dev), None)

    ths = [threading.Thread(target=worker, args=(d,), name=f"job@{d}", daemon=True) for d in devices]
    for t in ths:
        t.start()
    last_hb, stalled = time.time(), set()
    try:
        while True:
            alive = [t for t in ths if t.is_alive()]
            if not alive:
                break
            alive[0].join(timeout=5.0)
            now = time.time()
            if now - last_hb >= 60 * cfg.heartbeat_min:
                last_hb = now
                busy = " | ".join(f"{d}: {v['name']} ep {v['ep']}/{cfg.epochs} ({(now - v['t_job']) / 60:.1f} ph)"
                                  for d, v in sorted(_PROG.items())) or "—"
                log(f"  ⏱ {done[0]}/{len(jobs)} xong sau {(now - t0) / 60:.0f} ph | đang chạy: {busy}")
            for d, v in list(_PROG.items()):
                if now - v["t"] > 60 * cfg.stall_min and (d, v["name"]) not in stalled:
                    stalled.add((d, v["name"]))
                    _dump_stacks(f"{d} không tiến triển {(now - v['t']) / 60:.0f} phút ở job {v['name']}")
    except KeyboardInterrupt:
        _STOP.set()
        log(f"[{title}] NGẮT: chờ các luồng dừng ở vòng lặp kế tiếp, lưu {len(RESULTS)} job đã xong...")
        for t in ths:
            t.join(timeout=120)
        save_results()
        raise
    save_results()
    for j in failed:                                   # chạy lại tuần tự một lần (kết quả đã xong đã được lưu)
        log(f"  ↻ chạy lại {_job_name(j)} trên {devices[0]}")
        r = train_job(j, devices[0]); RESULTS[r["key"]] = r
        save_results()
    log(f"[{title}] xong trong {(time.time() - t0) / 60:.1f} phút")

def _bn_layers(m):
    return [x for x in m.modules() if isinstance(x, nn.modules.batchnorm._BatchNorm)]

def _reestimate_bn(m, tgt_tr, device):
    # ước lượng lại thống kê BatchNorm bằng dữ liệu đích không nhãn (trung bình tích luỹ, không augmentation)
    for b in _bn_layers(m):
        b.reset_running_stats(); b.momentum = None
    m.eval()
    for b in _bn_layers(m):
        b.train()
    with torch.no_grad():
        idx = np.random.default_rng(0).permutation(np.asarray(tgt_tr))
        for s in range(0, len(idx), 128):
            with amp_autocast(device, amp_on(device)):
                m(fetch(idx[s:s + 128], device))
    return m.eval()

def adabn_model(state, tgt_tr, device):
    # AdaBN (Li et al., 2018): giữ nguyên trọng số source-only, chỉ ước lượng lại thống kê BatchNorm.
    return _reestimate_bn(load_state(LeakNet().to(device), state), tgt_tr, device)

def tent_model(state, tgt_tr, device, seed=0):
    # [V2] TENT (Wang et al., ICLR 2021): thích ứng LÚC KIỂM THỬ, không cần dữ liệu nguồn — chỉ cập nhật tham số
    # affine (γ, β) của BatchNorm bằng cực tiểu entropy dự đoán trên đích không nhãn (thống kê theo lô), sau đó
    # ước lượng lại thống kê BN để suy luận tất định. Sát với triển khai: mô hình gửi đi, tự thích ứng tại điểm mới.
    m = adabn_model(state, tgt_tr, device)
    bns = _bn_layers(m)
    for q in m.parameters():
        q.requires_grad_(False)
    params = [q for b in bns for q in (b.weight, b.bias) if q is not None]
    for q in params:
        q.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=cfg.tent_lr)
    rng = np.random.default_rng(seed)
    m.eval()
    for b in bns:
        b.train(); b.momentum = 0.1
    tgt_tr = np.asarray(tgt_tr)
    for _ in range(cfg.tent_steps):
        xb = fetch(rng.choice(tgt_tr, min(cfg.batch_size, len(tgt_tr)), replace=False), device)
        loss = binary_entropy(m(xb)).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    for q in m.parameters():
        q.requires_grad_(False)
    return _reestimate_bn(m, tgt_tr, device)

POSTHOC_BUILDERS = {"adabn": lambda st, tr, dev, sd: adabn_model(st, tr, dev),
                    "tent": lambda st, tr, dev, sd: tent_model(st, tr, dev, sd)}

def run_posthoc(variant="main", methods=None):
    methods = [m for m in (methods or POSTHOC_METHODS) if m in cfg.methods]
    n = 0
    for T in TASKS:
        for s in cfg.seeds:
            src = RESULTS.get(job_key(dict(variant=variant, task=T["i"], seed=s, fold=None, method="source_only")))
            if src is None or src["state"] is None:
                continue
            P = PARTS[(T["i"], s)]
            for meth in methods:
                for k in range(len(P["folds"])):
                    j = dict(variant=variant, task=T["i"], seed=s, fold=k, method=meth)
                    if job_key(j) in RESULTS:
                        continue
                    t0 = time.time()
                    tgt_tr, tgt_te = P["fold_idx"][k]
                    m = POSTHOC_BUILDERS[meth](src["state"], tgt_tr, DEVICE, 1000 * s + k)
                    p = infer(m, tgt_te, DEVICE); del m
                    RESULTS[job_key(j)] = dict(key=job_key(j), variant=variant, task=T["i"], seed=s, fold=k,
                                               method=meth, shots=0, hp={}, hist=None, ep_hist=None,
                                               te_idx=np.asarray(tgt_te), sel=dict(src["sel"]),
                                               pred={r: p for r in RULES}, thr=dict(src["thr"]),
                                               src_val_rec_auc=dict(src["src_val_rec_auc"]),
                                               state=None, n_skip=0, time_s=time.time() - t0, device=str(DEVICE))
                    n += 1
    save_results()
    log(f"Hậu kiểm {methods}: {n} lượt (từ mô hình source-only của từng seed).")

run_adabn = lambda variant="main": run_posthoc(variant, ["adabn"])
print("Job runner OK")
''')

# =============================================================================
# 8. CHẠY THÍ NGHIỆM CHÍNH
# =============================================================================
md(r'''
## 8. Thí nghiệm chính / Main experiment

`bài toán × seed × fold × phương pháp`. `source_only` chạy một lần mỗi seed (giữ trọng số cho AdaBN), các phương pháp
còn lại chạy theo từng fold đích. Trọng số (seed đầu, fold 0) được giữ cho phân tích không gian đặc trưng ở mục 11.

**Theo dõi một lượt chạy dài (v2.1).** Ô này chạy nhiều giờ (≈ 1.5–2 phút/job/GPU, ≈ 5–6 giờ cho cấu hình mặc định).
- Mỗi job xong in một dòng; **luồng chính in nhịp tim `⏱` mỗi `cfg.heartbeat_min` phút** (job + epoch đang chạy trên
  từng GPU). Nếu một GPU không tiến triển quá `cfg.stall_min` phút, stack trace của mọi luồng được in ra để chẩn đoán.
- Mọi dòng tiến độ đồng thời ghi vào `run_log.txt` (thư mục output). Ở **phiên tương tác**, nếu trình duyệt mất kết
  nối thì ô **ngừng hiện output dù kernel vẫn chạy** — đó không phải treo. Nên chạy bằng **Save Version → Save & Run
  All (Commit)**: lượt chạy diễn ra phía máy chủ, xem tiến độ ở tab *Logs* của phiên bản.
- `ada_results.pkl` được lưu mỗi `cfg.save_every_min` phút. Chạy lại ô (cùng phiên) hoặc thêm output của phiên bản
  trước làm **Input** (phiên mới) -> chỉ các job còn thiếu được chạy. Có thể chạy theo giai đoạn: `seeds=(42,)` trước
  (≈ 1/3 thời gian), sau đó `seeds=(42, 43, 44)`.
- Ngắt ô (■ Interrupt) dừng sạch cả hai luồng GPU ở vòng lặp kế tiếp và lưu các job đã xong.
''')

code(r'''
# =========================================================
# 8.1 Danh sách job + chạy / build job list and run
# =========================================================
TRAINED = [m for m in cfg.methods if m in ADAPT_METHODS + ("oracle", "target_only")]
NEED_SO = ("source_only" in cfg.methods) or any(m in cfg.methods for m in POSTHOC_METHODS)

def main_jobs(variant="main", methods=None, seeds=None, tasks=None, hp=None, shots=0):
    methods = TRAINED if methods is None else methods
    seeds = cfg.seeds if seeds is None else seeds
    tasks = [T["i"] for T in TASKS] if tasks is None else tasks
    out = []
    for t in tasks:
        for s in seeds:
            for k in range(len(PARTS[(t, s)]["folds"])):
                for m in methods:
                    j = dict(variant=variant, task=t, seed=s, fold=k, method=m,
                             keep_state=(variant == "main" and s == cfg.seeds[0] and k == 0))
                    if hp:
                        j["hp"] = hp
                    if shots:
                        j["shots"] = shots
                    out.append(j)
    return out

MAIN_JOBS = []
if NEED_SO:
    MAIN_JOBS += [dict(variant="main", task=T["i"], seed=s, fold=None, method="source_only", keep_state=True)
                  for T in TASKS for s in cfg.seeds]
MAIN_JOBS += main_jobs()
print(f"Thí nghiệm chính: {len(MAIN_JOBS)} job huấn luyện "
      f"({len(TASKS)} bài toán × {len(cfg.seeds)} seed; phương pháp {list(cfg.methods)})")
run_jobs(MAIN_JOBS, "main")
run_posthoc("main")

if cfg.save_models:
    for r in RESULTS.values():
        if r["variant"] == "main" and r.get("state") is not None:
            torch.save(r["state"], CKPT_DIR / f"{TASKS[r['task']]['slug']}_{r['method']}_s{r['seed']}_f{r['fold']}.pt")

_tm = pd.DataFrame([dict(method=r["method"], time_s=r["time_s"], device=r["device"], n_skip=r["n_skip"])
                    for r in RESULTS.values() if r["variant"] == "main"])
cost = _tm.groupby("method").agg(n_jobs=("time_s", "size"), mean_s=("time_s", "mean"),
                                 total_min=("time_s", lambda s: s.sum() / 60),
                                 skipped_steps=("n_skip", "sum")).reset_index()
save_table(cost, "training_cost", "Training cost per method (wall-clock per job and total; skipped non-finite AMP steps).", 1)
''')
