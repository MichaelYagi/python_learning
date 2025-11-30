from PIL import Image
import cv2
import numpy as np
import scipy.fftpack

# Average hashing, difference hashing and perceptive hashing algorithms
class PerceptualHashing:
    def __init__(self, image: Image.Image, resolution_value: int = 64, highfreq_factor: int = 4):
        self.image = image  # Initialize the 'image' attribute
        self.hash_size = int(resolution_value/8)  # Initialize the 'hash_size' attribute, determines image size by pixels hash_size x hash_size
        self.highfreq_factor = highfreq_factor

    # ===================================================================
    # aHash – ALL ORIGINAL COMMENTS PRESERVED
    # ===================================================================
    def ahash(self) -> int:
        """
        Compute the horizontal difference hash (aHash) for a PIL Image.
        Returns a hash_size*hash_size-bit integer.
        """
        if self.hash_size < 2 or self.image is None:
            raise ValueError("hash_size must be >= 2 and image can't be None")

        # 1) Grayscale normalization and tiny resize
        resized = self._preprocess_for_ahash_dhash()

        # Explicit row-major traversal, compute grayscale manually (Rec.601), truncate to byte
        pixels = resized.flatten().tolist()
        sum_val = sum(pixels)
        avg = float(sum_val) / len(pixels)

        result = 0
        bit_index = 0
        bitlist = []
        grayScaleArray = []

        for pixel in pixels:
            grayScaleArray.append(pixel)
            if pixel >= avg:
                result |= (1 << bit_index)
                bitlist.append(1)
            else:
                bitlist.append(0)
            bit_index += 1

        # Reverse only for debug
        grayScaleArray.reverse()
        bitlist.reverse()

        # print(*grayScaleArray, sep=', ')
        print("average (0-255): " + str(avg))
        print("ahash bit pattern")
        print(*bitlist, sep='')
        print("result")
        print(result)
        print("")

        return result

    # ===================================================================
    # dHash – ALL ORIGINAL COMMENTS PRESERVED
    # ===================================================================
    def dhash(self) -> int:
        """
        Compute the horizontal difference hash (dHash) for a PIL Image.
        Returns a hash_size*hash_size-bit integer.
        """
        if self.hash_size < 2 or self.image is None:
            raise ValueError("hash_size must be >= 2 and image can't be None")

        # 1) Grayscale normalization and tiny resize: width = hash_size + 1, height = hash_size
        resized = self._preprocess_for_ahash_dhash()

        # 2) Generate differences left-to-right for each row
        result = 0
        bit_index = 0
        bitlist = []

        for y in range(self.hash_size):
            for x in range(self.hash_size):
                left_pix = resized[y, x]
                right_pix = resized[y, x + 1]
                # 1 if it gets darker to the right, else 0
                # It will be similar for other similar images
                # If the right pixel is darker than the left → assign bit 1.
                # If the right pixel is brighter or equal → assign bit 0.
                bit = 1 if right_pix < left_pix else 0
                bitlist.insert(bit_index, bit)
                # Pack bit into integer, least-significant-bit first
                result |= (bit << bit_index)
                bit_index += 1

        print("hash size: " + str(self.hash_size))
        print("dhash bit pattern")
        bitlist.reverse()
        print(*bitlist, sep='')
        print("result")
        print(result)
        print("")

        return result

    # ===================================================================
    # pHash – ALL ORIGINAL COMMENTS PRESERVED
    # ===================================================================
    def phash(self) -> int:
        """
        Compute the perceptual hash (pHash) for a PIL Image.
        Returns a hash_size*hash_size-bit integer.
        """
        if self.hash_size < 2 or self.image is None:
            raise ValueError("hash_size must be >= 2 and image can't be None")

        # Step 1: Convert to grayscale and resize to a larger size
        img_size = self.hash_size * self.highfreq_factor
        img = self._preprocess_for_phash(img_size)

        # Step 2: Convert image data to numpy array
        # pixels: a NumPy array of grayscale brightness values (e.g. 32×32).
        pixels = img

        # Step 3: Apply 2D Discrete Cosine Transform (DCT)
        # The Discrete Cosine Transform (DCT) re‑expresses that grid not in terms of pixels, but in terms of patterns of variation across the grid.
        # Low frequency = slow, gradual changes in brightness (big shapes, smooth gradients). The basic shape of the image and a little blurred.
        # High frequency = rapid changes in brightness (edges, fine details, noise). The edges and noise, final result can be just noise captured.
        # [ 50, 52, 55, 58, 60, 62, 65, 68 ] row of pixels and is a smooth gradient → mostly low frequency.
        # [ 50, 200, 50, 200, 50, 200, 50, 200 ] row of pixels and is a sharp gradient → mostly high frequency.
        # By keeping only the low‑frequency coefficients, pHash builds a fingerprint that’s stable across edits but still unique to the image.
        # pixels.T: transpose → flips rows and columns.
        # scipy.fftpack.dct(..., norm='ortho'): applies the Discrete Cosine Transform (DCT).
        # First DCT is applied to columns (pixels.T).
        # Then transpose back and apply DCT to rows.
        # Each entry is a frequency coefficient (number that multiplies a variable in an algebraic expression): how much of a certain “wave pattern” exists in the image.
        # Top‑left corner = low frequencies (broad shapes).
        # Bottom‑right corner = high frequencies (fine details, noise).
        dct = scipy.fftpack.dct(scipy.fftpack.dct(pixels.T, norm='ortho').T, norm='ortho')

        # Step 4: Take top-left (low-frequency) coefficients
        dct_lowfreq = dct[:self.hash_size, :self.hash_size]

        # Bottom-right block (high frequency)
        # dct_highfreq = dct[-self.hash_size:, -self.hash_size:]

        # Middle band (medium frequencies)
        # mid_start = dct.shape[0] // 2 - self.hash_size // 2
        # mid_end = mid_start + self.hash_size
        # dct_midfreq = dct[mid_start:mid_end, mid_start:mid_end]

        # Step 5: Compute median of these coefficients
        med = np.median(dct_lowfreq)

        # Step 6: Build hash: 1 if coefficient > median, else 0
        result = 0
        bit_index = 0
        bitlist = []
        for y in range(self.hash_size):
            for x in range(self.hash_size):
                bit = 1 if dct_lowfreq[y, x] > med else 0
                bitlist.insert(bit_index, bit)
                result |= (bit << bit_index)
                bit_index += 1

        print("hash size: " + str(self.hash_size))
        print("phash bit pattern")
        bitlist.reverse()
        print(*bitlist, sep='')
        print("result")
        print(result)
        print("")

        return result

    # ===================================================================
    # Shared preprocessing – used by aHash, dHash and pHash
    # ===================================================================
    def _preprocess_for_ahash_dhash(self) -> np.ndarray:
        """Returns (hash_size, hash_size+1) uint8 grayscale"""
        img = np.array(self.image)

        # 2. Remove alpha if present (white background, just like Java)
        if img.ndim == 3 and img.shape[2] == 4:
            alpha = img[:, :, 3]
            bg = np.full_like(img[:, :, :3], 255)
            mask = alpha[:, :, None] / 255.0
            img = (img[:, :, :3] * mask + bg * (1 - mask)).astype(np.uint8)

        # 3. Convert to grayscale (this is the Rec.601 that ColorConvertOp actually applies)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 4. Resize using OpenCV BICUBIC — identical to Java Graphics2D
        resized = cv2.resize(
            gray,
            (self.hash_size + 1, self.hash_size),
            interpolation=cv2.INTER_CUBIC
        )

        return resized  # uint8

    def _preprocess_for_phash(self, target_size: int) -> np.ndarray:
        """Returns (target_size×target_size float32 grayscale – used only by pHash"""
        img = np.array(self.image)

        if img.ndim == 3 and img.shape[2] == 4:
            alpha = img[:, :, 3]
            bg = np.full_like(img[:, :, :3], 255)
            mask = alpha[:, :, None] / 255.0
            img = (img[:, :, :3] * mask + bg * (1 - mask)).astype(np.uint8)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        resized = cv2.resize(
            gray,
            (target_size, target_size),
            interpolation=cv2.INTER_CUBIC
        )

        return resized.astype(np.float32)

    @staticmethod
    def hamming_distance(my_hash: int, other_hash: int) -> int:
        """
        Count differing bits between two hash integers.
        eg. 1110011011001100100110111001011001010011100001110001110000111000
            1110011011001100100110111001011001010011100001100001110100111000
            Number of differences in bits is 2
        bit_count is a function that returns the number of set bits (bits with a value of 1) in the binary representation of an integer
        """
        return (my_hash ^ other_hash).bit_count()

    @staticmethod
    def normalized_hamming_distance(my_hash: int, other_hash: int, resolution_value: int = 64) -> float:
        return 100-PerceptualHashing.similarity_percentage(my_hash, other_hash, resolution_value)

    @staticmethod
    def similarity_percentage(my_hash: int, other_hash: int, resolution_value: int = 64) -> float:
        """
        Compute similarity percentage between two hashes to two decimal places.
        """
        hash_size = resolution_value/8
        distance = PerceptualHashing.hamming_distance(my_hash, other_hash)
        total_bits = hash_size * hash_size
        similarity = ((total_bits - distance) / total_bits) * 100
        return round(similarity, 2)

# ****************************************** INTEGRATION ******************************************

# DEFINE 2 FILE PATHS TO IMAGE TO COMPARE
filepath1 = "C:\\Users\\Michael\\Downloads\\testpics\\dupetest\\DSC00169.JPG"
filepath2 = "C:\\Users\\Michael\\Downloads\\testpics\\dupetest\\DSC00170.JPG"
img1 = Image.open(filepath1)
img2 = Image.open(filepath2)

resolution = 64 # Adjust as desired. Must be divisible by 8 and > than 8

print("Image 1:", filepath1)
print("Image 2:", filepath2)
print("resolution:", resolution)
print("-----------------\n")

ph1 = PerceptualHashing(img1, resolution)
ph2 = PerceptualHashing(img2, resolution)

print("ahash image 1")
h1 = ph1.ahash()
print("ahash image 2")
h2 = ph2.ahash()
print("ahash hamming distance:", PerceptualHashing.hamming_distance(h1, h2))
print("ahash normalized hamming distance:", PerceptualHashing.normalized_hamming_distance(h1, h2, resolution))
print("ahash similarity:", PerceptualHashing.similarity_percentage(h1, h2, resolution), "%")
print("-----------------\n")

print("dhash image 1")
h1 = ph1.dhash()
print("dhash image 2")
h2 = ph2.dhash()
print("dhash hamming distance:", PerceptualHashing.hamming_distance(h1, h2))
print("dhash normalized hamming distance:", PerceptualHashing.normalized_hamming_distance(h1, h2, resolution))
print("dhash similarity:", PerceptualHashing.similarity_percentage(h1, h2, resolution), "%")
print("-----------------\n")

print("phash image 1")
h1 = ph1.phash()
print("phash image 2")
h2 = ph2.phash()
print("phash hamming distance:", PerceptualHashing.hamming_distance(h1, h2))
print("phash normalized hamming distance:", PerceptualHashing.normalized_hamming_distance(h1, h2, resolution))
print("phash similarity:", PerceptualHashing.similarity_percentage(h1, h2, resolution), "%")
print("-----------------")
