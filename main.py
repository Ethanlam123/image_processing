import numpy as np
from ImageCompressor import *

# Initialize image compressor with an image file
compressor = ImageCompressor("lena.jpg")

# Test SVD compression, saving and PSNR calculation
k = 100
print("SVD compressed image:")
plt.imshow(compressor.SVDcompress(k))
plt.show()

print("Saving compressed image...")
compressor.save_compressed_image(k)

print("PSNR of the compressed image:", compressor.calculate_psnr(k))

# Test norm calculation
print("Norm of the original and compressed image:", compressor.norm(k))

# Test image comparison
print("Comparison of original and compressed images:")
compressor.compare_images(k)

# Test compression ratio calculation
print("Compression ratio:", compressor.compression_ratio(k))

# Test grayscale application
print("Grayscale image:")
compressor.apply_grayscale()

# Test adding noise
print("Noisy image:")
noisy_img = compressor.add_noise(mean=0, std=1)
plt.imshow(noisy_img)
plt.show()

# Test DCT compression
print("DCT compressed image:")
dct_compressed_img = compressor.DCTcompress(k)
plt.imshow(dct_compressed_img)
plt.show()

# Test PCA compression
print("PCA compressed image:")
pca_compressed_img = compressor.PCAcompress(k)
plt.imshow(pca_compressed_img)
plt.show()

# Test saving compressed image
print("Saving compressed image...")
compressor.save_compressed_image(k)
