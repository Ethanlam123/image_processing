import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.fftpack import dct, idct


class ImageCompressor:
    def __init__(self, img_path):
        self.img_path = img_path
        self.image_array = np.asarray(Image.open(self.img_path))
        self.SVD = {'r': None, 'g': None, 'b': None}

        self.compressed_image = None
        self.k = None

    def rSVD(self, X, r, q, p):
        # random projection
        ny = X.shape[1]
        P = np.random.randn(ny, r + p)
        Z = X @ P
        for k in range(q):
            Z = X @ (X.T @ Z)
        # qr decomposition
        Q, R = np.linalg.qr(Z, mode='reduced')
        Y = Q.T @ X
        UY, S, VT = np.linalg.svd(Y, full_matrices=0)
        U = Q @ UY

        return U, S, VT
    
    def compute_svd(self, k):
        if self.SVD['r'] is None:
            for i, color in enumerate(('r', 'g', 'b')):
                self.SVD[color] = self.rSVD(self.image_array[:, :, i], k, 1, 5)

    def _compress(self, k):
        if self.compressed_image is None or self.k != k:
            self.compute_svd(k)
            self.k = k
            rimg = np.zeros_like(self.image_array)
            for i, color in enumerate(('r', 'g', 'b')):
                U, S, VT = self.SVD[color]
                rimg[:, :, i] = np.dot(U[:, :k], np.dot(np.diag(S[:k]), VT[:k, :]))
            rimg = np.clip(rimg, 0, 255)
            self.compressed_image = rimg.astype(np.uint8)
        return self.compressed_image

    def SVDcompress(self, k):
        return self._compress(k)
    
    def plot_singular_values(self):
        self.compute_svd(100)

        for color, label in zip(('r', 'g', 'b'), ('Red', 'Green', 'Blue')):
            _, S, _ = self.SVD[color]
            plt.plot(S[:100], label=label)
            
        plt.legend()
        plt.title('Singular Values')
        plt.show()

    def plot_compressed_image(self, k):
        compressed_image = self.SVDcompress(k)
        plt.imshow(compressed_image)
        plt.show()
    
    def plot_original_image(self):
        plt.imshow(self.image_array)
        plt.show()
    
    def norm(self, k):
        compressed_image = self._compress(k)
        return np.linalg.norm(self.image_array - compressed_image)

    def save_compressed_image(self, k):
        compressed_image = self._compress(k)
        Image.fromarray(compressed_image).save(f"rSVD_{self.img_path}_k{k:04}.jpg")

    def compare_images(self, k):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        ax[0].imshow(self.image_array)
        ax[0].set_title('Original Image')
        ax[0].axis('off')

        compressed_image = self.SVDcompress(k)
        ax[1].imshow(compressed_image)
        ax[1].set_title('Compressed Image')
        ax[1].axis('off')

        plt.show()

    def compression_ratio(self, k):
        original_size = np.prod(self.image_array.shape)
        compressed_size = k * (1 + sum(self.image_array.shape))
        return original_size / compressed_size

    def apply_grayscale(self):
        gray = np.dot(self.image_array[...,:3], [0.2989, 0.5870, 0.1140])
        plt.imshow(gray, cmap=plt.get_cmap('gray'))
        plt.show()

    def add_noise(self, mean=0, std=1):
        # add Gaussian noise to image
        noisy_img = self.image_array + np.random.normal(mean, std, self.image_array.shape)
        noisy_img_clipped = np.clip(noisy_img, 0, 255)
        return noisy_img_clipped.astype(np.uint8)

    def calculate_psnr(self, k):
        compressed_image = self._compress(k)
        mse = np.mean((self.image_array - compressed_image) ** 2)
        PIXEL_MAX = 255.0
        return np.where(mse == 0, 100, 20 * np.log10(PIXEL_MAX / np.sqrt(mse)))

    def dct2(self, block):
        return dct(dct(block.T, norm='ortho').T, norm='ortho')

    def idct2(self, block):
        return idct(idct(block.T, norm='ortho').T, norm='ortho')

    def DCTcompress(self, k):
        imsize = self.image_array.shape
        dct_img = np.zeros(imsize)

        for i in range(3):  
            dct_img[:,:,i] = self.dct2(self.image_array[:,:,i])

        # Threshold
        dct_img[:,:,0] = np.multiply(dct_img[:,:,0], np.abs(dct_img[:,:,0]) > k)
        dct_img[:,:,1] = np.multiply(dct_img[:,:,1], np.abs(dct_img[:,:,1]) > k)
        dct_img[:,:,2] = np.multiply(dct_img[:,:,2], np.abs(dct_img[:,:,2]) > k)

        compressed_img = np.zeros(imsize)
        for i in range(3):
            compressed_img[:,:,i] = self.idct2(dct_img[:,:,i])
            
        return compressed_img

    def PCAcompress(self, k):
        img = self.image_array
        original_shape = img.shape
        img_flattened = img.reshape(-1, original_shape[-1])  # Flatten image

        # Normalize the data
        img_normalized = img_flattened - np.mean(img_flattened, axis=0)
        
        # Compute the covariance matrix
        cov_matrix = np.cov(img_normalized.T)

        # Compute the eigenvectors and eigenvalues of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Sort eigenvalues and corresponding eigenvectors
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        # Select the first k eigenvectors
        eigenvectors_subset = sorted_eigenvectors[:, :k]

        # Transform the data to the first k eigenvectors
        img_transformed = np.dot(img_normalized, eigenvectors_subset)

        # Reconstruct the image
        img_reconstructed = np.dot(img_transformed, eigenvectors_subset.T) + np.mean(img_flattened, axis=0)

        # Reshape image to original shape
        img_reconstructed_reshaped = img_reconstructed.reshape(original_shape)

        return img_reconstructed_reshaped.astype(np.uint8)
