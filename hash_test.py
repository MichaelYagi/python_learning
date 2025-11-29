from perceptual_hashing import PerceptualHash
from PIL import Image

# DEFINE 2 FILE PATHS TO IMAGE TO COMPARE
img1 = Image.open("<image_path_1>")
img2 = Image.open("<image_path_2>")

resolution = 32
ph1 = PerceptualHash(img1, resolution)
ph2 = PerceptualHash(img2, resolution)

h1 = ph1.ahash()
h2 = ph2.ahash()
print("ahash hamming distance:", PerceptualHash.hamming_distance(h1, h2))
print("ahash similarity:", PerceptualHash.similarity_percentage(h1, h2), "%")
print("-----------------")

h1 = ph1.dhash()
h2 = ph2.dhash()
print("dhash hamming distance:", PerceptualHash.hamming_distance(h1, h2))
print("dhash similarity:", PerceptualHash.similarity_percentage(h1, h2), "%")
print("-----------------")

h1 = ph1.phash()
h2 = ph2.phash()
print("phash hamming distance:", PerceptualHash.hamming_distance(h1, h2))
print("phash himilarity:", PerceptualHash.similarity_percentage(h1, h2), "%")
print("-----------------")