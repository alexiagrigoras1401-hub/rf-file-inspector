from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict

import numpy as np
import scipy.signal as sps
from scipy.io import wavfile


def load_wav(path: str) -> Tuple[np.ndarray, float, bool]:
    fs, data = wavfile.read(path)
    if data.ndim > 1:
        data = data[:, 0]
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / float(np.iinfo(data.dtype).max)
    else:
        data = data.astype(np.float32)
    return data.astype(np.float32), float(fs), False


def load_iq(path: str, dtype: str, fs: float) -> Tuple[np.ndarray, float, bool]:
    dt = {"int8": np.int8, "int16": np.int16, "float32": np.float32}.get(dtype)
    if dt is None:
        raise ValueError(f"dtype invalid: {dtype}")

    raw = np.fromfile(path, dtype=dt)
    if raw.size < 2:
        raise ValueError("Fișier IQ prea mic.")
    raw = raw[: (raw.size // 2) * 2]

    I = raw[0::2].astype(np.float32)
    Q = raw[1::2].astype(np.float32)

    if np.issubdtype(dt, np.integer):
        maxv = float(np.iinfo(dt).max)
        I /= maxv
        Q /= maxv

    x = (I + 1j * Q).astype(np.complex64)
    return x, float(fs), True


def normalize_power(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.complex64, copy=False)
    p = np.sqrt(np.mean(np.abs(x) ** 2) + 1e-12)
    return x / p


def welch_psd_db(x: np.ndarray, fs: float, nfft: int = 8192):
    onesided = not np.iscomplexobj(x)
    f, Pxx = sps.welch(
        x,
        fs=fs,
        nperseg=nfft,
        noverlap=nfft // 2,
        detrend=False,
        return_onesided=onesided,
        scaling="density",
    )
    psd_db = 10.0 * np.log10(np.maximum(Pxx, 1e-20))

    if np.iscomplexobj(x):
        f = np.fft.fftshift(f)
        psd_db = np.fft.fftshift(psd_db)
    return f, psd_db


def robust_noise_floor(psd_db: np.ndarray) -> float:
    return float(np.median(psd_db))


def spectral_flatness(psd_lin: np.ndarray) -> float:
    psd_lin = np.maximum(psd_lin, 1e-30)
    g = np.exp(np.mean(np.log(psd_lin)))
    a = np.mean(psd_lin)
    return float(g / (a + 1e-30))


def occupied_bw(freqs: np.ndarray, psd_db: np.ndarray, floor_db: float, thresh_db: float = 10.0) -> float:
    mask = psd_db > (floor_db + thresh_db)
    if not np.any(mask):
        return 0.0
    fmin = float(freqs[mask][0])
    fmax = float(freqs[mask][-1])
    return float(abs(fmax - fmin))


def narrowband_peakiness(psd_db: np.ndarray, floor_db: float) -> float:
    return float(np.max(psd_db) - floor_db)


def instantaneous_freq_stats(x: np.ndarray, fs: float) -> Dict[str, float]:
    if not np.iscomplexobj(x):
        x = sps.hilbert(x).astype(np.complex64)

    x = normalize_power(x)
    ph = np.unwrap(np.angle(x))
    dph = np.diff(ph)
    inst_f = (fs / (2 * np.pi)) * dph
    inst_f = inst_f[np.isfinite(inst_f)]
    if inst_f.size < 10:
        return {"if_std": 0.0, "if_kurt": 0.0}

    clip = np.percentile(inst_f, [1, 99])
    inst_fc = inst_f[(inst_f >= clip[0]) & (inst_f <= clip[1])]
    if inst_fc.size < 10:
        inst_fc = inst_f

    std = float(np.std(inst_fc))
    m = np.mean(inst_fc)
    v = np.mean((inst_fc - m) ** 2) + 1e-12
    k = float(np.mean((inst_fc - m) ** 4) / (v * v) - 3.0)
    return {"if_std": std, "if_kurt": k}


def amplitude_stats(x: np.ndarray) -> Dict[str, float]:
    a = np.abs(x).astype(np.float32) if np.iscomplexobj(x) else np.abs(x.astype(np.float32))
    a = a / (np.sqrt(np.mean(a**2)) + 1e-12)
    std = float(np.std(a))
    p95 = float(np.percentile(a, 95))
    p50 = float(np.percentile(a, 50))
    env_ratio = float(p95 / (p50 + 1e-12))
    return {"amp_std": std, "env_ratio": env_ratio}


def constellation_metrics(x: np.ndarray) -> Dict[str, float]:
    if not np.iscomplexobj(x):
        x = sps.hilbert(x).astype(np.complex64)
    x = normalize_power(x)

    k = max(1, x.size // 4096)
    if k > 1:
        b = sps.firwin(129, 1.0 / k * 0.8)
        xf = sps.lfilter(b, [1.0], x)
        xs = xf[::k]
    else:
        xs = x

    xs = xs[np.isfinite(xs)]
    if xs.size < 200:
        return {"radius_std": 0.0, "phase_std": 0.0, "iq_kurt": 0.0}

    r = np.abs(xs)
    ph = np.angle(xs)
    radius_std = float(np.std(r))
    phase_std = float(np.std(np.unwrap(ph)))

    I = np.real(xs)
    Q = np.imag(xs)

    def ekurt(v):
        m = np.mean(v)
        vv = np.mean((v - m) ** 2) + 1e-12
        return float(np.mean((v - m) ** 4) / (vv * vv) - 3.0)

    iq_kurt = 0.5 * (ekurt(I) + ekurt(Q))
    return {"radius_std": radius_std, "phase_std": phase_std, "iq_kurt": iq_kurt}


def classify_jamming(freqs, psd_db, floor_db):
    psd_lin = 10 ** (psd_db / 10)
    flat = spectral_flatness(psd_lin)
    peakiness = narrowband_peakiness(psd_db, floor_db)
    obw = occupied_bw(freqs, psd_db, floor_db, thresh_db=10.0)

    jtype = "none"
    jammed = False

    if peakiness > 25.0:
        jtype = "narrowband_interference"
        jammed = True
    else:
        if flat > 0.35 and obw > 0.2 * (freqs[-1] - freqs[0]):
            jtype = "wideband_noise"
            jammed = True

    feats = {"flatness": float(flat), "peakiness_db": float(peakiness), "occupied_bw_hz": float(obw)}
    return jtype, jammed, feats


def classify_signal_type(x: np.ndarray, fs: float, psd_db: np.ndarray, floor_db: float):
    amp = amplitude_stats(x)
    inst = instantaneous_freq_stats(x, fs)
    const = constellation_metrics(x)

    env_ratio = amp["env_ratio"]
    amp_std = amp["amp_std"]
    if_std = inst["if_std"]
    radius_std = const["radius_std"]
    iq_kurt = const["iq_kurt"]
    carrier_peak = float(np.max(psd_db) - floor_db)

    features = {**amp, **inst, **const, "carrier_peak_db": carrier_peak}

    if env_ratio > 1.6 and amp_std > 0.35:
        return "AM", features

    if if_std > 0.02 * fs:
        return "FM", features

    if abs(iq_kurt) < 0.7 and radius_std > 0.35 and carrier_peak < 25.0:
        return "OFDM (simulated)", features

    if radius_std < 0.28 and amp_std < 0.25:
        return "PSK", features

    if radius_std >= 0.28:
        return "QAM", features

    return "Unknown", features


@dataclass
class Report:
    file: str
    fs_hz: float
    center_hz: Optional[float]
    n_samples: int
    is_iq: bool
    signal_type: str
    jammed: bool
    jamming_type: str
    noise_floor_db: float
    diagnostics: Dict[str, float]

    def to_json_bytes(self) -> bytes:
        return json.dumps(asdict(self), indent=2).encode("utf-8")


def analyze_path(path: str, kind: str, fs: Optional[float], dtype: str, center_hz: Optional[float] = None) -> Tuple[Report, Tuple[np.ndarray, np.ndarray]]:
    if kind == "wav":
        x, fs2, _ = load_wav(path)
        fs = fs2
    else:
        if fs is None:
            raise ValueError("fs is required for IQ")
        x, fs2, _ = load_iq(path, dtype, fs)
        fs = fs2
        x = normalize_power(x)

    # Use up to ~2 seconds for stability
    max_n = int(min(x.size, fs * 2.0))
    x = x[:max_n]

    freqs, psd_db = welch_psd_db(x, fs, nfft=8192)
    floor_db = robust_noise_floor(psd_db)

    jtype, jammed, jfeats = classify_jamming(freqs, psd_db, floor_db)
    stype, sfeats = classify_signal_type(x, fs, psd_db, floor_db)

    diag = {"noise_floor_db": float(floor_db), **jfeats, **sfeats}
    rep = Report(
        file=path,
        fs_hz=float(fs),
        center_hz=float(center_hz) if center_hz is not None else None,
        n_samples=int(x.size),
        is_iq=bool(np.iscomplexobj(x)),
        signal_type=stype,
        jammed=bool(jammed),
        jamming_type=jtype,
        noise_floor_db=float(floor_db),
        diagnostics=diag,
    )
    return rep, (freqs, psd_db)
