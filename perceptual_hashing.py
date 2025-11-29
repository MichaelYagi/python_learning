import time
from ftplib import print_line

from PIL import Image
import numpy as np
import scipy.fftpack

# Average hashing, difference hashing and perceptive hashing algorithms
class PerceptualHashing:
    def __init__(self, image: Image.Image, hash_size: int = 8, highfreq_factor: int = 4):
        self.image = image  # Initialize the 'image' attribute
        self.hash_size = hash_size  # Initialize the 'hash_size' attribute, determines image size by pixels hash_size x hash_size
        self.highfreq_factor = highfreq_factor

    def set_hash_size(self, hash_size: int) -> None:
        self.hash_size = hash_size

    def ahash(self) -> int:
        """
        Compute the horizontal difference hash (dHash) for a PIL Image.
        Returns a hash_size*hash_size-bit integer.
        """
        if self.hash_size < 2 or self.image == None:
            raise ValueError("hash_size must be >= 2 and image can't be None")

        # 1) Grayscale normalization and tiny resize: width = hash_size + 1, height = hash_size
        img = self.image.convert("L").resize(
            (self.hash_size + 1, self.hash_size),
            Image.Resampling.LANCZOS
        )

        # filename = f"C:/Users/Michael/Downloads/py_test_image_{int(time.time())}.jpg"
        # img.save(filename, format="JPEG")

        # 2) Compute average pixel brightness value
        # If the image is grayscale ("L" mode) → each element is a shade of grey integer (0–255).
        pixels = list(img.getdata()) # type: ignore
        # length is (self.hash_size + 1) * self.hash_size
        avg = sum(pixels) / len(pixels)

        result = 0
        bit_index = 0
        bitlist = []

        for pixel in pixels:
            bit = 1 if pixel >= avg else 0
            bitlist.insert(bit_index, bit)
            # If it’s brighter than the average → assign bit 1.
            # If it’s darker than the average → assign bit 0.
            result |= (bit << bit_index)
            bit_index += 1

        print("pixels: "+str(len(pixels)))
        print("average (0-255): " + str(avg))
        print("ahash bit pattern")
        bitlist.reverse()
        print(*bitlist, sep='')
        print("result")
        print(result)
        print("")

        return result

    def dhash(self) -> int:
        """
        Compute the horizontal difference hash (dHash) for a PIL Image.
        Returns a hash_size*hash_size-bit integer.
        """
        if self.hash_size < 2 or self.image == None:
            raise ValueError("hash_size must be >= 2 and image can't be None")

        # 1) Grayscale normalization and tiny resize: width = hash_size + 1, height = hash_size
        img = self.image.convert("L").resize(
            (self.hash_size + 1, self.hash_size),
            Image.Resampling.LANCZOS
        )

        # filename = f"C:/Downloads/py_test_image_{int(time.time())}.jpg"
        # img.save(filename, format="JPEG")

        # 2) Generate differences left-to-right for each row
        pixels = img.load()
        result = 0
        bit_index = 0
        bitlist = []

        for y in range(self.hash_size):
            for x in range(self.hash_size):
                left_pix = pixels[x, y]
                right_pix = pixels[x + 1, y]
                # 1 if it gets darker to the right, else 0
                # It will be similar for other similar images
                # If the right pixel is darker than the left → assign bit 1.
                # If the right pixel is brighter or equal → assign bit 0.
                bit = 1 if right_pix < left_pix else 0
                bitlist.insert(bit_index, bit)
                # print(left_pix, end=", ")
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

    def phash(self) -> int:
        """
        Compute the perceptual hash (pHash) for a PIL Image.
        Returns a hash_size*hash_size-bit integer.
        """
        if self.hash_size < 2 or self.image is None:
            raise ValueError("hash_size must be >= 2 and image can't be None")

        # Step 1: Convert to grayscale and resize to a larger size
        img_size = self.hash_size * self.highfreq_factor
        img = self.image.convert("L").resize(
            (img_size, img_size),
            Image.Resampling.LANCZOS
        )

        # filename = f"C:/Users/Michael/Downloads/py_test_image_{int(time.time())}.jpg"
        # img.save(filename, format="JPEG")

        # Step 2: Convert image data to numpy array
        # pixels: a NumPy array of grayscale brightness values (e.g. 32×32).
        pixels = np.array(img, dtype=np.float32)

        # Step 3: Apply 2D Discrete Cosine Transform (DCT)
        # The Discrete Cosine Transform (DCT) re‑expresses that grid not in terms of pixels, but in terms of patterns of variation across the grid.
        # Low frequency = slow, gradual changes in brightness (big shapes, smooth gradients).
        # High frequency = rapid changes in brightness (edges, fine details, noise).
        # [ 50, 52, 55, 58, 60, 62, 65, 68 ] row of pixels and is a smooth gradient → mostly low frequency.
        # [ 50, 200, 50, 200, 50, 200, 50, 200 ] row of pixels and is a sharp gradient → mostly high frequency.
        # Low frequencies capture the overall structure of the image (shapes, layout).
        # High frequencies are easily changed by noise, compression, or small edits.
        # By keeping only the low‑frequency coefficients, pHash builds a fingerprint that’s stable across edits but still unique to the image.
        # pixels.T: transpose → flips rows and columns.
        # scipy.fftpack.dct(..., norm='ortho'): applies the Discrete Cosine Transform (DCT).
        # First DCT is applied to columns (pixels.T).
        # Then transpose back and apply DCT to rows.
        # Each entry is a frequency coefficient: how much of a certain “wave pattern” exists in the image.
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
    def similarity_percentage(my_hash: int, other_hash: int, hash_size: int = 8) -> float:
        """
        Compute similarity percentage between two hashes to two decimal places.
        """
        distance = PerceptiveHash.hamming_distance(my_hash, other_hash)
        total_bits = hash_size * hash_size
        similarity = ((total_bits - distance) / total_bits) * 100
        return round(similarity, 2)