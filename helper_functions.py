import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import torch
from scipy.stats import norm
import seaborn as sns

# Helper functions for the project

# Plot activation functions
def plot_activation_functions(functions, title, x_range = (-5, 5), num_points = 500):
    """
    Plots one or multiple activation functions on the same plot.
    """
    # If a single function is passed, convert it to a list with one tuple
    if not isinstance(functions, list):
        functions = [functions]

    # Generate input values
    x = torch.linspace(x_range[0], x_range[1], num_points)
    
    # Initialize the plot
    plt.figure(figsize = (10, 6))
    
    # Plot each activation function
    for func, func_name in functions:
        y = func(x)  # Compute outputs
        plt.plot(x.numpy(), y.numpy(), label = func_name, linewidth = 2)

    # Add plot details
    plt.axhline(0, color = "black", linewidth = 0.5, linestyle = "--")
    plt.axvline(0, color = "black", linewidth = 0.5, linestyle = "--")
    plt.title(title, fontsize = 16)
    plt.xlabel("Input", fontsize = 14)
    plt.ylabel("Output", fontsize = 14)
    plt.legend(fontsize = 12)
    plt.grid(alpha = 0.3)
    plt.show()

# Plot normal distrubition
def plot_normal_distribution(x):
    """
    Plots the normal distribution with areas up to x and after x in different colors.
    """
    # Generate x values for the plot (from -4 to 4 standard deviations)
    x_values = np.linspace(-4, 4, 1000)
    
    # Calculate the standard normal distribution (mean = 0, std = 1)
    y_values = norm.pdf(x_values, 0, 1)
    
    # Create the plot
    plt.figure(figsize = (8, 6))
    
    # Plot the full normal distribution curve
    plt.plot(x_values, y_values, label = "Standard Normal Distribution", color = "black")
    
    # Fill the area under the curve up to the input x with one color (light blue)
    plt.fill_between(x_values, y_values, where = (x_values <= x), color = "lightblue", label = "Area up to x", alpha = 0.6)
    
    # Fill the rest of the area (beyond x) with another color (light coral)
    plt.fill_between(x_values, y_values, where = (x_values > x), color = "lightcoral", label = "Area beyond x", alpha = 0.6)
    
    # Highlight the input x as a vertical line
    plt.axvline(x = x, color = "red", linestyle = "--", label = f"x = {x}")
    
    # Add labels and title
    plt.title("Standard Normal Distribution and area up to x")
    plt.xlabel("x")
    plt.ylabel("Probability Density")
    
    # Show the legend
    plt.legend(loc = "upper left")
    
    # Display the plot
    plt.show()

# Show image split into patches
def show_image_split_into_patches(image, patch_size):
    """
    Shows an image split into patches.
    """
    # Convert image to numpy array
    image_array = np.array(image)

    # Create patches list
    patch_size = patch_size
    patches = []

    # Loop to extract 16x16 patches
    for i in range(0, image_array.shape[0], patch_size):
        for j in range(0, image_array.shape[1], patch_size):
            patch = image_array[i: i  + patch_size, j : j + patch_size]
            patches.append(patch)

    # Visualize the image patches using matplotlib
    fig, axes = plt.subplots(14, 14, figsize = (10, 10))

    # Flatten the axes array for easy indexing  
    axes = axes.flatten()

    for i, patch in enumerate(patches):
        axes[i].imshow(patch)
        axes[i].axis('off')  # Hide axes

    plt.tight_layout()
    plt.show()

# Plot class distribution for a specific dataset
def plot_class_distribution(split_dist, split_name, class_names):
    """ 
    Plots the class distribution for a classification dataset.
    """
    # Create a DataFrame from the class distribution
    data = {
        "Class": class_names,
        "Count": [split_dist.get(i, 0) for i in range(len(class_names))]
    }

    df = pd.DataFrame(data)

    # Create a barplot using seaborn
    plt.figure(figsize = (12, 6))
    sns.barplot(x = "Class", y = "Count", data = df)

    # Rotate x-axis ticks for better readability and adjust font size
    plt.xticks(rotation = 90, fontsize = 10)  # Rotate and adjust font size

    # Set a maximum number of x-ticks to avoid overlap
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer = True, prune = "both"))  # Adjust tick placement

    # Set title and labels
    plt.title(f"Class Distribution in {split_name} Split")
    plt.xlabel("Class")
    plt.ylabel("Count")

    # Display the plot
    plt.show()

