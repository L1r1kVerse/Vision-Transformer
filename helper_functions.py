import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import norm

# Helper functions for the project

# Plot activation functions
def plot_activation_functions(functions, title, x_range = (-5, 5), num_points = 500):
    """
    Plots one or multiple activation functions on the same graph.
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