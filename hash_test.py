from perceptive_hash import PerceptiveHash
from PIL import Image

# DEFINE 2 FILE PATHS TO IMAGE TO COMPARE
img1 = Image.open("<image_path_1>")
img2 = Image.open("<image_path_2>")

resolution = 32
ph1 = PerceptiveHash(img1, resolution)
ph2 = PerceptiveHash(img2, resolution)

h1 = ph1.ahash()
h2 = ph2.ahash()
print("ahash hamming distance:", PerceptiveHash.hamming_distance(h1, h2))
print("ahash similarity:", PerceptiveHash.similarity_percentage(h1, h2), "%")
print("-----------------")

h1 = ph1.dhash()
h2 = ph2.dhash()
print("dhash hamming distance:", PerceptiveHash.hamming_distance(h1, h2))
print("dhash similarity:", PerceptiveHash.similarity_percentage(h1, h2), "%")
print("-----------------")

h1 = ph1.phash()
h2 = ph2.phash()
print("phash hamming distance:", PerceptiveHash.hamming_distance(h1, h2))
print("phash himilarity:", PerceptiveHash.similarity_percentage(h1, h2), "%")
print("-----------------")