import os
import numpy as np
from scipy.fft import idct
from scipy.linalg import orth
from skimage import color, transform, io
import h5py
import random

# Function for reconstructing the image from compressed measurements
def cs_reconstruction(input_image, Phi, Theta_inv, psi, index, save_folder, m, noise_level):
    # Input Image
    im_size = input_image.shape[0]
    A = input_image
    x = A.flatten().astype(float)
    n = len(x)
    
    # Compression
    y = np.dot(Phi, x)
    
    # L2 norm solution
    s2 = np.dot(Theta_inv, y)
    
    # Image Reconstruction
    x2 = np.zeros(n)
    for ii in range(n):
        x2 += np.dot(psi[ii, :, :].T, s2[ii])
    
    recons_image = np.reshape(x2, (im_size, im_size))
    
    # Save reconstructed image
    name = f"recons_image_{index}.png"
    image_name = os.path.join(save_folder, name)
    io.imsave(image_name, np.uint8(recons_image))
    
    return recons_image

# Preprocess function to resize and convert images to grayscale
def preprocess_stl10(data_folder, dataset, save_folder, ratio, noise_level):
    # Load raw data
    print('-- Loading raw data')
    file_path = os.path.join(data_folder, f"{dataset}.mat")
    with h5py.File(file_path, 'r') as f:
        X = np.array(f['X'])
    
    print('-- Preprocessing')
    
    # Reshape to 96x96x3
    X_im = np.reshape(X, (X.shape[0], 96, 96, 3))
    
    # Init output arrays
    X_resized = np.zeros((X_im.shape[0], 64, 64), dtype=np.uint8)
    
    for jj in range(X_im.shape[0]):
        # Convert to grayscale and resize
        grayscale_image = color.rgb2gray(X_im[jj, :, :, :])
        resized_image = transform.resize(grayscale_image, (64, 64), anti_aliasing=True)
        X_resized[jj, :, :] = (resized_image * 255).astype(np.uint8)
    
    # Compression ratio
    compression_ratio = ratio
    X_recons = np.zeros_like(X_resized)
    
    # Number of measurements
    im_size = X_resized.shape[1] * X_resized.shape[2]
    n = im_size
    m = int(compression_ratio * n)
    
    # Measurement matrix Phi
    Phi = np.random.randn(m, n)
    Phi = orth(Phi.T).T
    
    # Theta matrix calculation
    Theta = np.zeros((m, n))
    Shi = np.zeros((n, n, 1))
    
    for ii in range(n):
        ek = np.zeros(n)
        ek[ii] = 1
        psi = idct(ek, norm='ortho')
        Theta[:, ii] = np.dot(Phi, psi)
        Shi[ii, :, :] = psi.reshape(-1, 1)
    
    # Theta inverse
    Theta_inv = np.linalg.pinv(Theta)
    
    # Save images after reconstruction
    for kk in range(45000):  # Adjust based on available images
        X_recons[kk, :, :] = cs_reconstruction(X_resized[kk, :, :], Phi, Theta_inv, Shi, kk, save_folder, m, noise_level)
    
    # Save as preprocessed images
    print('-- Saving preprocessed data')
    output_file = os.path.join(save_folder, f"{dataset}_CS_reconstructed.npz")
    np.savez(output_file, X_recons=X_recons)
    print(f"Data saved to {output_file}")

# Main function to process the dataset
def main():
    # User-defined paths
    data_folder = os.path.join('data', 'stl10_matlab')
    save_folder = '/path_to_save/reconstructed_images/'
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # Set compression ratio and noise level
    compression_ratio = 0.4
    noise_level = 1e-4
    
    # Preprocess training data (for 'unlabeled' dataset)
    preprocess_stl10(data_folder, 'unlabeled', save_folder, compression_ratio, noise_level)

if __name__ == "__main__":
    main()
