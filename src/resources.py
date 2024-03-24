import json
import matplotlib.pyplot as plt
import numpy as np

class Resources:
    """A class containing some usefull functions that are not directly related to the problems in
    task 1 and 2.
    """
    
    @staticmethod
    def get_saved_plot(name: str) -> dict:
        """
        Load a saved dictionary of conductances versus iterations from a JSON file.

        Parameters:
            name (str): Name of the file to be loaded. Do not include path, just filename
            without the .json suffix.

        Returns:
            dict: A dictionary containing conductances and their corresponding iterations.
        """
        # Load the plot dictionary from the JSON file
        with open(f"Results/saved_plots/{name}.json", "r") as json_file:
            plot = json.load(json_file)
        
        return plot

    @staticmethod
    def plot_conductances(calculated_values: dict, normalized: bool = True) -> None:
        """
        Plot conductances versus iterations as well as mean, standard deviation and variance.

        Parameters:
            plot (dict): A dictionary containing conductances and their corresponding iterations.
            
        Returns:
            None
        """
        # Define font properties
        font = {
            'family': 'serif', 
            'color': 'darkred', 
            'weight': 'normal', 
            'size': 16
        }

        # Make the plot
        plt.figure(figsize=(18, 7))
        
        # Set xlabel, ylabel and title of the plot
        plt.xlabel(
            'Iterations', 
            fontsize=30, 
            fontdict=font
        )
        plt.ylabel(
            'Conductance [$2\\frac{e^2}{h}$]' 
            if normalized else 
            "Conductance [$\\Omega^{-1}$]", 
            fontsize=30, 
            fontdict=font
        )
        plt.title(
            f'Conductance pr. iteration - $\\alpha$ = 0', 
            fontsize=35, 
            fontdict=font, 
            y=1.05
        )

        # Plot the mean as a black line
        plt.axhline(
            calculated_values["normalized mean"]
            if normalized else
            calculated_values["mean"], 
            color = 'green', 
            linestyle = '--', 
            label = 'Mean',
            linewidth = 1
        )
        # Plot standard deviation and variance as opaque regions
        plt.axhspan(
            calculated_values["normalized mean"] - calculated_values["normalized standard deviation"]
            if normalized else
            calculated_values["mean"] - calculated_values["standard deviation"], 
            calculated_values["normalized mean"] + calculated_values["normalized standard deviation"]
            if normalized else
            calculated_values["mean"] + calculated_values["standard deviation"], 
            color = 'blue', 
            alpha = 0.2, 
            label = 'Standard Deviation'
        )
        plt.axhspan(
            calculated_values["normalized mean"] - calculated_values["normalized variance"]
            if normalized else
            calculated_values["mean"] - calculated_values["variance"], 
            calculated_values["normalized mean"] + calculated_values["normalized variance"]
            if normalized else
            calculated_values["mean"] + calculated_values["variance"], 
            color = 'red', 
            alpha = 0.3, 
            label = 'Variance'
        )

        # Plot the conductance values
        plt.scatter(
            calculated_values["iterations"], 
            calculated_values["normalized conductances"]
            if normalized else
            calculated_values["conductances"], 
            label="Conductance", 
            color="black",
            s = 7
        )

        # Set some parameters
        plt.ylim(29.5, 30.5)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend(fontsize=20)
        
        # Show plot
        plt.show()
 
    @staticmethod
    def visualize_matrix(mat: np.ndarray) -> None:
        """
        Visualize a matrix.

        This method plots the absolute values of the input matrix.

        Args:
            mat (np.ndarray): The input matrix.
        """
        
        plt.figure(figsize=(8, 6))  # Adjust figure size
        
        plt.imshow(np.abs(mat), cmap='viridis')
        plt.colorbar(label='Transmission probability')
        plt.title("$\\hat{T}_{tot_{nm}}$ for $\\alpha = 0.035$", fontsize=18)  # Increase title font size
        plt.xlabel('$n$', fontsize=14)  # Add x-axis label with font size
        plt.ylabel('$m$', fontsize=14)  # Add y-axis label with font size
        plt.xticks(fontsize=12)  # Increase tick label font size
        plt.yticks(fontsize=12)  # Increase tick label font size
        plt.grid(True, linewidth = 1)  # Add gridlines
        plt.grid(False)
        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()

    @staticmethod
    def visualize_matrix_of_matrices(matrices: np.ndarray, title: str) -> None:
        """
        Visualize a matrix of matrices.

        This method plots a matrix of matrices, where each element of the input
        matrices array is a matrix itself.

        Args:
            matrices (np.ndarray): The input matrix of matrices.
            title (str): The title of the plot.
        """
        
        subplot_titles = [
            ['$t_{tot}$', "$r'_{tot}$"],
            ["$r_{tot}$", "$t'_{tot}$"]
        ]
        
        num_rows, num_cols = matrices.shape[0], matrices.shape[1]
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 8))
        
        for i in range(num_rows):
            for j in range(num_cols):
                im = axs[i, j].imshow(np.abs(matrices[i, j]), cmap='viridis')
                axs[i, j].axis('off')
                axs[i, j].set_title(subplot_titles[i][j])
                cbar = fig.colorbar(im, ax=axs[i, j], label="Magnitude")
                cbar.ax.yaxis.get_offset_text().set_fontsize(5)
                
        fig.suptitle(title)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_matrix_of_matrices_complex(matrices: np.ndarray, title: str) -> None:
        """
        Visualize a matrix of complex matrices.

        This method plots a matrix of matrices, where each element of the input
        matrices array is a complex matrix itself.

        Args:
            matrices (np.ndarray): The input matrix of complex matrices.
            title (str): The title of the plot.
        """
        subplot_titles = [
            ['$t_{tot}$', "$r'_{tot}$"],
            ["$r_{tot}$", "$t'_{tot}$"]
        ]
        
        num_rows, num_cols = matrices.shape[0], matrices.shape[1]
        fig, axs = plt.subplots(num_rows, num_cols * 2, figsize=(15, 30))
        
        
        for i in range(num_rows):
            for j in range(num_cols):
                # Plot real part
                im_real = axs[i, 2*j].imshow(np.real(matrices[i, j]), cmap='viridis')
                axs[i, 2*j].axis('off')
                cbar_r = fig.colorbar(im_real, ax=axs[i, 2*j], label="Real Part", shrink=0.45)
                cbar_r.ax.yaxis.get_offset_text().set_fontsize(5)
                
                # Plot imaginary part
                im_imag = axs[i, 2*j+1].imshow(np.imag(matrices[i, j]), cmap='viridis')
                axs[i, 2*j+1].axis('off')
                cbar_i = fig.colorbar(im_imag, ax=axs[i, 2*j+1], label="Imaginary Part", shrink=0.45)
                cbar_i.ax.yaxis.get_offset_text().set_fontsize(5)
                
                # Set subplot titles
                axs[i, 2*j].set_title(f"$Re${{{subplot_titles[i][j]}}}")
                axs[i, 2*j+1].set_title(f"$Im${{{subplot_titles[i][j]}}}")
        
        fig.suptitle(title)
        plt.tight_layout()
        plt.show()
        
 