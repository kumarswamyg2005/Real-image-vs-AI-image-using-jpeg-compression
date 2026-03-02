import cv2
import numpy as np
from scipy.fftpack import dct
import heapq
from collections import Counter

# -----------------------------
# 1. COLOR TRANSFORM (RGB → YCbCr)
# -----------------------------
def rgb_to_ycbcr(img):
    img = img.astype(np.float32)

    Y  =  0.299*img[:,:,2] + 0.587*img[:,:,1] + 0.114*img[:,:,0]
    Cb = -0.1687*img[:,:,2] - 0.3313*img[:,:,1] + 0.5*img[:,:,0] + 128
    Cr =  0.5*img[:,:,2] - 0.4187*img[:,:,1] - 0.0813*img[:,:,0] + 128

    return Y, Cb, Cr



# -----------------------------
# 2. DOWN-SAMPLING (4:2:0)
# -----------------------------
def downsample(channel):
    return channel[::2, ::2]


# -----------------------------
# 3. BLOCK SPLITTING (8×8)
# -----------------------------
def split_blocks(channel):
    h, w = channel.shape
    blocks = []

    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = channel[i:i+8, j:j+8]

            if block.shape == (8, 8):
                blocks.append(block)

    return blocks


# -----------------------------
# 4. FORWARD DCT
# -----------------------------
def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


# -----------------------------
# 5. QUANTIZATION
# -----------------------------
Q = np.array([
 [16,11,10,16,24,40,51,61],
 [12,12,14,19,26,58,60,55],
 [14,13,16,24,40,57,69,56],
 [14,17,22,29,51,87,80,62],
 [18,22,37,56,68,109,103,77],
 [24,35,55,64,81,104,113,92],
 [49,64,78,87,103,121,120,101],
 [72,92,95,98,112,100,103,99]
])

def quantize(block):
    return np.round(block / Q).astype(int)


# -----------------------------
# 6. ZIGZAG SCAN
# -----------------------------
def zigzag(block):
    h, w = block.shape
    result = []

    for s in range(h + w - 1):  #s = diagonal index.
        if s % 2 == 0:  #Even diagonal → upward
            for i in range(s+1):
                j = s - i
                if i < h and j < w:
                    result.append(block[i][j])
        else:
            for i in range(s+1):
                j = s - i
                if j < h and i < w:
                    result.append(block[j][i])

    return result


# -----------------------------
# 7. HUFFMAN ENCODING
# -----------------------------
class Node:
    def __init__(self, symbol, freq):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman(data):
    freq = Counter(data)
    heap = [Node(sym, f) for sym, f in freq.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        n1 = heapq.heappop(heap)
        n2 = heapq.heappop(heap)

        merged = Node(None, n1.freq + n2.freq)
        merged.left = n1
        merged.right = n2
        heapq.heappush(heap, merged)

    return heap[0]


def generate_codes(node, prefix="", codebook={}):
    if node is None:
        return

    if node.symbol is not None:
        codebook[node.symbol] = prefix

    generate_codes(node.left, prefix+"0", codebook)
    generate_codes(node.right, prefix+"1", codebook)

    return codebook


def huffman_encode(data):
    tree = build_huffman(data)
    codebook = generate_codes(tree)

    encoded = "".join(codebook[symbol] for symbol in data)

    return encoded, codebook


# -----------------------------
# MAIN JPEG COMPRESSION
# -----------------------------
def jpeg_compress(image_path):

    # Read image
    img = cv2.imread(image_path)

    # Resize to multiple of 16
    h, w, _ = img.shape
    img = img[:h - h%16, :w - w%16]

    # 1. Color Transform
    Y, Cb, Cr = rgb_to_ycbcr(img)

    # 2. Downsample chroma
    Cb_ds = downsample(Cb)
    Cr_ds = downsample(Cr)

    channels = [Y, Cb_ds, Cr_ds]
    all_coeffs = []

    for ch in channels:

        # Level shift
        ch = ch - 128

        # 3. Split into 8×8 blocks
        blocks = split_blocks(ch)

        for block in blocks:

            # 4. Forward DCT
            dct_block = dct2(block)

            # 5. Quantization
            q_block = quantize(dct_block)

            # 6. Zigzag
            zz = zigzag(q_block)

            all_coeffs.extend(zz)

    # 7. Huffman Encoding → Compressed Data
    bitstream, codebook = huffman_encode(all_coeffs)

    return bitstream, codebook


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    import sys
    
    # Use command-line argument if provided, otherwise use test image
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "test_sample/re7.JPG"
    
    print(f"Compressing: {image_path}")
    compressed_data, codes = jpeg_compress(image_path)

    print("Compressed Bitstream Length:", len(compressed_data))
    print("Sample Bitstream (first 500 bits):")
    print(compressed_data[:500])