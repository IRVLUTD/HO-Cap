import matplotlib.pyplot as plt
import numpy as np
import os


def plot_and_save_images(images):
    """
    Plot the images in the specified layout and save as 1080P PNG.

    Parameters:
    images (list of numpy arrays): List of 10 images to be displayed.
    frame_id (int): The frame ID to be used in the filename.
    output_folder (str): The folder where the output images will be saved.
    """
    if len(images) != 10:
        raise ValueError("The function expects exactly 10 images.")

    # Create a figure with 1920x1080 resolution
    fig = plt.figure(
        figsize=(19.2, 10.8), dpi=100
    )  # figsize in inches, dpi=100 for 1920x1080 pixels

    # Create a GridSpec with 3 rows and 4 columns
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.5])

    # Plot the first 8 images in a 2x4 grid
    for i in range(8):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        ax.imshow(images[i])
        ax.axis("off")  # Hide the axes

    # Plot the 9th image on the bottom left
    ax = fig.add_subplot(gs[2, :2])
    ax.imshow(images[8])
    ax.axis("off")  # Hide the axes

    # Plot the 10th image on the bottom right
    ax = fig.add_subplot(gs[2, 2:])
    ax.imshow(images[9])
    ax.axis("off")  # Hide the axes

    # Save the figure with 1920x1080 resolution
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Display the plot
    plt.tight_layout()
    plt.show()


# # Example usage with random images
output_folder = "output_images"  # Adjust this path
# for frame_id in range(10):  # Assume 10 frames
#     images = [np.random.rand(480, 640, 3) for _ in range(8)]  # 8 images of size 640x480
#     images.append(np.random.rand(720, 1280, 3))  # 1 image of size 1280x720
#     plot_and_save_images(images, frame_id, output_folder)


images = [np.random.rand(480, 640, 3) for _ in range(8)]  # 8 images of size 640x480
images.append(np.random.rand(720, 1280, 3))  # 1 image of size 1280x720
images.append(np.random.rand(720, 1280, 3))  # 1 image of size 1280x720
plot_and_save_images(images)
