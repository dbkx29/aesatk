import numpy as np
from collections import defaultdict


def recover_permutation(plain_img, scrambled_img):
    """
    已知明文攻击（Known-Plaintext Attack）恢复 permutation。

    plain_img      : 原图像（攻击者拥有）
    scrambled_img  : 仅经过 permutation 后的图像（攻击者从 AES 解密得到）
    """

    flat_plain = plain_img.reshape(-1, plain_img.shape[-1])
    flat_scrambled = scrambled_img.reshape(-1, scrambled_img.shape[-1])

    # 记录【某个像素值】在 plain 中出现的所有位置
    pos_map = defaultdict(list)
    for idx, px in enumerate(map(tuple, flat_plain.tolist())):
        pos_map[px].append(idx)

    # 用于保存 permutation
    perm = np.full(len(flat_plain), -1, dtype=np.int64)

    # 对 scrambled 的每个像素，找到它在 plain 中可能的原位置
    for i, px in enumerate(map(tuple, flat_scrambled.tolist())):
        lst = pos_map[px]
        if lst:
            perm[i] = lst.pop()
        else:
            perm[i] = -1  # 像素重复导致无法精确定位

    return perm


def apply_inverse_permutation(scrambled_img, perm):
    """
    使用逆 permutation 还原图像。
    """
    h, w, c = scrambled_img.shape
    flat = scrambled_img.reshape(-1, c)
    restored = np.zeros_like(flat)

    for scrambled_idx, plain_idx in enumerate(perm):
        if plain_idx != -1:
            restored[plain_idx] = flat[scrambled_idx]

    return restored.reshape(h, w, c)


def demo():
    # 生成示例图片（10x10、RGB）
    np.random.seed(42)
    plain = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)

    # 模拟真实系统中的 permutation
    permutation = np.random.permutation(plain.size // 3)
    scrambled = plain.reshape(-1, 3)[permutation].reshape(10, 10, 3)

    # === 攻击者场景：攻击者已知 plain，并从 AES 解密得到 scrambled ===
    recovered_perm = recover_permutation(plain, scrambled)
    restored = apply_inverse_permutation(scrambled, recovered_perm)

    print("Recovered pixels ratio:",
          np.mean(restored == plain) * 100, "%")

    print("Sample recovered permutation (first 20):")
    print(recovered_perm[:20])


if __name__ == "__main__":
    demo()
