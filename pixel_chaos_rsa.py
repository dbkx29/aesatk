"""
pixel_chaos_rsa.py

pixel_chaos_rsa:
  1) 读取图像 -> 得到 raw pixels (H,W,C)
  2) 使用 chaos_seed 派生 RNG seed -> 用 Fisher-Yates 做可逆置乱 (O(n))
  3) 将置乱后的 raw bytes 使用 AES-GCM 加密 (ephemeral AES key per file)
  4) 使用 RSA-OAEP 加密该 ephemeral AES key（混合加密）
  5) 输出二进制文件，包含: MAGIC, VERSION, MODE, nonce(12), rsa_key_len(2), enc_key, shape(12), ct_len(4), ciphertext, tag(16)
"""

import os
import struct
import time
import hashlib
import numpy as np
import cv2
from Crypto.Cipher import AES, PKCS1_OAEP
from Crypto.PublicKey import RSA
from typing import Tuple

MAGIC = b'PCR1'   # Pixel Chaos RSA v1
VERSION = 1
MODE_PIXEL_CHAOS_RSA = 2

# ---------------- helpers ----------------
def sha256_bytes(s: str) -> bytes:
    return hashlib.sha256(s.encode()).digest()

def derive_seed_from_password(password: str) -> int:
    """从密码派生一个 64-bit 的整数用于 RNG seed（可调整位数）"""
    digest = sha256_bytes(password)
    return int.from_bytes(digest[:8], 'big')

def generate_rsa_keypair(bits=2048) -> Tuple[bytes, bytes]:
    key = RSA.generate(bits)
    return key.publickey().export_key(), key.export_key()

# ---------- permutation (Fisher-Yates) ----------
def make_permutation(n: int, seed: int) -> np.ndarray:
    """生成长度 n 的置换 (permutation array of indices) 使用 Fisher-Yates。
       返回 perm: np.ndarray(shape=(n,), dtype=np.int64) 表示 new_index = perm[old_index]
       我们保持定义： scrambled[i] = flat[ perm[i] ]  （perm 给出从原到新）
    """
    rng = np.random.default_rng(seed)  # 使用 PCG / numpy 非线性 RNG
    perm = np.arange(n, dtype=np.int64)
    # Fisher-Yates shuffle
    for i in range(n - 1, 0, -1):
        j = rng.integers(0, i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    return perm

def invert_permutation(perm: np.ndarray) -> np.ndarray:
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm), dtype=np.int64)
    return inv

# # ---------- pixel scramble/descramble ----------
# def scramble_pixels(arr: np.ndarray, seed_int: int) -> np.ndarray:
#     """arr: (H,W,C) uint8  -> 返回同 shape 的 scrambled arr"""
#     h, w = arr.shape[:2]
#     if arr.ndim == 2:
#         c = 1
#         flat = arr.reshape(-1, 1)
#     else:
#         c = arr.shape[2]
#         flat = arr.reshape(-1, c)
#     n = flat.shape[0]
#     perm = make_permutation(n, seed_int)
#     scrambled_flat = flat[perm]
#     scrambled = scrambled_flat.reshape((h, w, c)) if c > 1 else scrambled_flat.reshape((h, w))
#     return scrambled
#
# def descramble_pixels(arr: np.ndarray, seed_int: int) -> np.ndarray:
#     h, w = arr.shape[:2]
#     if arr.ndim == 2:
#         c = 1
#         flat = arr.reshape(-1, 1)
#     else:
#         c = arr.shape[2]
#         flat = arr.reshape(-1, c)
#     n = flat.shape[0]
#     perm = make_permutation(n, seed_int)
#     inv = invert_permutation(perm)
#     restored_flat = np.empty_like(flat)
#     restored_flat[inv] = flat
#     restored = restored_flat.reshape((h, w, c)) if c > 1 else restored_flat.reshape((h, w))
#     return restored

def scramble_pixels(arr, seed_int):
    h, w = arr.shape[:2]
    c = 1 if arr.ndim == 2 else arr.shape[2]
    flat = arr.reshape(-1, c)

    perm = make_permutation(len(flat), seed_int)
    scrambled_flat = flat[perm]

    return scrambled_flat.reshape((h, w, c)) if c > 1 else scrambled_flat.reshape((h, w))

def descramble_pixels(arr, seed_int):
    h, w = arr.shape[:2]
    c = 1 if arr.ndim == 2 else arr.shape[2]
    flat = arr.reshape(-1, c)

    perm = make_permutation(len(flat), seed_int)

    # 正确构造逆置换
    inv = np.empty_like(perm)
    for new_i, old_i in enumerate(perm):
        inv[old_i] = new_i

    restored_flat = flat[inv]
    return restored_flat.reshape((h, w, c)) if c > 1 else restored_flat.reshape((h, w))

# ---------- file format & encrypt/decrypt ----------
def encrypt_pixel_chaos_rsa(image_path: str, out_path: str, password_for_chaos: str, rsa_pub_pem: bytes) -> dict:
    """执行 pixel_chaos_rsa 加密，返回元信息"""
    # 1. read image as raw pixels
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("无法读取图像: " + image_path)
    if img.ndim == 2:
        img = img[:, :, None]  # H,W,1
    h, w, c = img.shape

    # 2. derive seed and scramble
    seed_int = derive_seed_from_password(password_for_chaos)  # 64-bit int
    scrambled = scramble_pixels(img, seed_int)

    # 3. bytes to encrypt
    raw_bytes = scrambled.tobytes()

    # 4. generate ephemeral AES key & AES-GCM encrypt
    aes_key = os.urandom(16)  # AES-128 ephemeral
    nonce = os.urandom(12)
    cipher = AES.new(aes_key, AES.MODE_GCM, nonce=nonce)
    ciphertext, tag = cipher.encrypt_and_digest(raw_bytes)

    # 5. RSA-OAEP encrypt ephemeral AES key
    rsa_pub = RSA.import_key(rsa_pub_pem)
    rsa_cipher = PKCS1_OAEP.new(rsa_pub)
    enc_key = rsa_cipher.encrypt(aes_key)
    enc_key_len = len(enc_key)
    # 6. write binary: MAGIC|VER|MODE|nonce(12)|enc_key_len(2)|enc_key|h(4)|w(4)|c(4)|ct_len(8)|ciphertext|tag(16)
    with open(out_path, 'wb') as f:
        f.write(MAGIC)
        f.write(struct.pack('B', VERSION))
        f.write(struct.pack('B', MODE_PIXEL_CHAOS_RSA))
        f.write(nonce)  # 12
        f.write(struct.pack('>H', enc_key_len))
        f.write(enc_key)
        f.write(struct.pack('>III', h, w, c))
        f.write(struct.pack('>Q', len(ciphertext)))
        f.write(ciphertext)
        f.write(tag)  # 16

    return {
        'orig_path': image_path,
        'out_path': out_path,
        'h': h, 'w': w, 'c': c,
        'enc_size': os.path.getsize(out_path),
        'time': None
    }

def decrypt_pixel_chaos_rsa(enc_path: str, out_image_path: str, password_for_chaos: str, rsa_priv_pem: bytes) -> dict:
    """解密 pixel_chaos_rsa"""
    data = open(enc_path, 'rb').read()
    p = 0
    assert data[p:p+4] == MAGIC; p += 4
    ver = data[p]; p += 1
    mode = data[p]; p += 1
    assert mode == MODE_PIXEL_CHAOS_RSA
    nonce = data[p:p+12]; p += 12
    enc_key_len = struct.unpack('>H', data[p:p+2])[0]; p += 2
    enc_key = data[p:p+enc_key_len]; p += enc_key_len
    h, w, c = struct.unpack('>III', data[p:p+12]); p += 12
    ct_len = struct.unpack('>Q', data[p:p+8])[0]; p += 8
    ciphertext = data[p:p+ct_len]; p += ct_len
    tag = data[p:p+16]; p += 16

    # RSA decrypt ephemeral key
    rsa_priv = RSA.import_key(rsa_priv_pem)
    rsa_cipher = PKCS1_OAEP.new(rsa_priv)
    aes_key = rsa_cipher.decrypt(enc_key)

    # AES-GCM decrypt
    cipher = AES.new(aes_key, AES.MODE_GCM, nonce=nonce)
    raw_bytes = cipher.decrypt_and_verify(ciphertext, tag)

    # reconstruct scrambled array and descramble
    # raw_bytes length = h*w*c
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    arr = arr.reshape((h, w, c)) if c > 1 else arr.reshape((h, w))
    seed_int = derive_seed_from_password(password_for_chaos)
    restored = descramble_pixels(arr, seed_int)

    # save to file: if c==1 save gray
    if c == 1:
        saved = restored.reshape((h, w))
    else:
        saved = restored
    cv2.imwrite(out_image_path, saved)

    return {'out_image_path': out_image_path, 'out_size': os.path.getsize(out_image_path)}

# ---------- PSNR utility ----------
def compute_psnr(img1_path: str, img2_path: str) -> float:
    a = cv2.imread(img1_path, cv2.IMREAD_UNCHANGED)
    b = cv2.imread(img2_path, cv2.IMREAD_UNCHANGED)
    if a is None or b is None:
        return -1.0
    # align channels
    if a.shape != b.shape:
        # try to fix single channel mismatch
        if a.ndim == 2 and b.ndim == 3:
            a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
        elif b.ndim == 2 and a.ndim == 3:
            b = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
        else:
            return -1.0
    return float(cv2.PSNR(a, b))

# ---------- example usage ----------
if __name__ == "__main__":
    # quick demo on ./test_images
    import glob, sys
    pub, priv = generate_rsa_keypair(2048)
    os.makedirs('pcr_out_1', exist_ok=True)
    password = "my_secret_password_for_chaos"

    for path in glob.glob('./test_images/*.jpg'):
        name = os.path.basename(path)
        enc = os.path.join('pcr_out_1', name + '.pcr.bin')
        dec = os.path.join('pcr_out_1', 'dec_' + name)
        t0 = time.time()
        encrypt_pixel_chaos_rsa(path, enc, password, pub)
        t1 = time.time()
        decrypt_pixel_chaos_rsa(enc, dec, password, priv)
        t2 = time.time()
        psnr = compute_psnr(path, dec)
        print(f"{name}: enc {os.path.getsize(enc)} bytes, dec {os.path.getsize(dec)} bytes, enc_time {(t1-t0):.3f}s dec_time {(t2-t1):.3f}s psnr {psnr:.2f}")
