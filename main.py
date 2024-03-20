from matplotlib import pyplot as plt
import math
import scipy
import numpy as np
import random
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt

class Homework2:
    
    scattering_points: np.ndarray
    distances: np.ndarray
    alpha: float
    k_m: List[float]
    p_n: List[np.ndarray]
    scattering_matrix: np.ndarray
    transfer_matrix: np.ndarray
    total_scattering_matrix: np.ndarray
    
    __slots__ = [
        'scattering_points', 'distances', 'alpha', 'k_m', 
        'p_n', 'scattering_matrix', 'transfer_matrix',
        "total_scattering_matrix"
    ]
    
    def __init__(self) -> None:
        
        
        self.scattering_points, self.distances = self.generate_scattering_points_and_free_space()
        
        # Define alpha
        self.alpha = 0
        
        # Generate the k_m vector
        self.k_m = self.generate_k_m()
        
        # Generate the 599 p_n matrices
        self.p_n = self.generate_p_n()

        # Generate the scattering matrix.
        self.scattering_matrix = self.generate_scattering_matrix()
        
        # Generate the transfer matrix from the scattering matrix.
        self.transfer_matrix = self.convert_S_to_M(self.scattering_matrix)
        
        # Generate the total scattering matrix
        self.total_scattering_matrix = self.calculate_total_scattering_matrix()
        
        # Visualize a matrix containing sub-matrices. Complex values are converted
        # to magnitude
        Functions.visualize_matrix_of_matrices(
            self.total_scattering_matrix,
            f"Total scattering matrix - $\\alpha = {self.alpha}$"
        )
        # Visualize a matrix containing sub-matrices. Real and imaginary
        # parts of the sub-matrices are plotted individually
        Functions.visualize_matrix_of_matrices_complex(
            self.total_scattering_matrix,
            f"Total scattering matrix matrix - $\\alpha = {self.alpha}$"
        )
            
    def calculate_total_scattering_matrix(self) -> np.ndarray:
        """
        Calculates the total scattering matrix.

        This method calculates the total scattering matrix by performing
        a series of matrix multiplications involving the transfer matrix
        and the previously generated p_n matrices.

        Returns:
            np.ndarray: The total scattering matrix.
        """
        p_n = self.p_n
        p_n.reverse()  # Reverse the order of p_n matrices for calculation
        
        subcalculation = None
        nr_calculations = 0
        
        for p in p_n:
            
            if nr_calculations == 0:
                # If first iteration, just add the last p matrix to subcalculation.
                subcalculation = p
            else: 
                # Matrix multiply the current p matrix with the product 
                # of the previous calculations.
                subcalculation = np.matmul(p, subcalculation)
                
            # Perform matrix multiplication with the transfer matrix and the
            # product of the previous calculations.
            subcalculation = np.matmul(self.transfer_matrix, subcalculation)
            
            nr_calculations += 1
        
        total_scattering_matrix = subcalculation
        
        # Return the total scattering matrix
        return total_scattering_matrix

    def generate_scattering_matrix(self) -> np.ndarray:
        """
        Generates a scattering matrix for a given set of parameters.

        Returns:
            np.ndarray: The scattering matrix, S, with shape (2, 2, 30, 30).
                S[0, 0] corresponds to the transmission coefficient t,
                S[0, 1] corresponds to the reflection coefficient r,
                S[1, 0] corresponds to the reflection coefficient r',
                and S[1, 1] corresponds to the transmission coefficient t'.
        """

        # Generate the matrix with complex entries
        matrix = (
            np.eye(60)  # Identity matrix with shape (60, 60)
          + (
                (np.exp(1j * self.alpha * 60) - 1)  # Complex exponential term
              / (60)  # Scalar division
            )
          * np.ones((60, 60))  # Array of ones with shape (60, 60)
        )
                
        # Extract submatrices from the main matrix
        t = matrix[:30, :30]  # Transmission coefficient t matrix
        rp = matrix[:30, -30:]  # Reflection coefficients r' matrix
        r = matrix[-30:, :30]  # Reflection coefficients r matrix
        tp = matrix[-30:, -30:]  # Transmission coefficients t' matrix
        
        # Construct the scattering matrix S
        S = np.array([
            [t, rp],  # Top-left block (t and r')
            [r, tp]   # Bottom-right block (r and t')
        ])
        
        return S
  
    @staticmethod
    def convert_S_to_M(S: np.ndarray) -> np.ndarray:
        """
        Converts a scattering matrix S to a transfer matrix M.

        Args:
            S (np.ndarray): The scattering matrix S with shape (2, 2, 30, 30).
                S[0, 0] corresponds to the transmission coefficient t,
                S[0, 1] corresponds to the reflection coefficient r',
                S[1, 0] corresponds to the reflection coefficient r,
                and S[1, 1] corresponds to the transmission coefficient t'.

        Returns:
            np.ndarray: The transfer matrix M with shape (2, 2, 30, 30).
                M[0, 0] corresponds to the transmission coefficients,
                M[0, 1] corresponds to the forward transfer coefficients,
                M[1, 0] corresponds to the reverse transfer coefficients,
                and M[1, 1] corresponds to the inverse transmission coefficients.
        """

        # Extract individual components from the scattering matrix
        t = S[0][0]    # Transmission coefficients
        tp = S[1][1]   # Inverse transmission coefficients
        r = S[1][0]    # Reverse transfer coefficients
        rp = S[0][1]   # Forward transfer coefficients
        
        # Calculate elements of the transfer matrix
        x1_y1 = (
            tp
            - np.matmul(
                r,
                np.matmul(
                    np.linalg.pinv(t),
                    rp                
                )
            )
        ) 
        x2_y1 = (
            np.matmul(
                r,
                np.linalg.pinv(t)
            )
        ) 
        x1_y2 = (
            -np.matmul(
                np.linalg.pinv(t),
                rp
            )
        )
        x2_y2 = (
            np.linalg.pinv(t)
        )
        
        # Construct the transfer matrix M
        M = np.array([
            [x1_y1, x2_y1],
            [x1_y2, x2_y2]
        ])
        
        return M
  
    def generate_scattering_points_and_free_space(self) -> Tuple[List[float], List[float]]:
        """
        Generates random coordinates for scattering points and distances of 
        free space between them.

        Returns:
            Tuple[List[float], List[float]]: A tuple containing two lists.
                The first list contains randomly generated coordinates.
                The second list contains the distances between consecutive coordinates.
        """

        coordinates = []  # Initialize an empty list to store coordinates
        distances = []    # Initialize an empty list to store distances
        
        # Generate random coordinates
        for i in range(600):
            coordinate = random.uniform(0, 60000)
            coordinates.append(coordinate)

        # Sort the coordinates in ascending order
        coordinates = np.array(coordinates)
        coordinates.sort()
        
        # Calculate distances between consecutive coordinates
        for index in range(1, len(coordinates)):
            distance = coordinates[index] - coordinates[index - 1]
            distances.append(distance)
        
        distances = np.array(distances)
        
        return coordinates, distances  # Return the generated coordinates and distances
             
    def generate_p_n(self) -> List[np.ndarray]:
        """
        Generates a list of matrices for a given set of distances.

        Returns:
            List[np.ndarray]: A list of matrices, where each matrix represents
                the transfer matrix corresponding to a specific distance.
                Each matrix has shape (2, 2, 30, 30) representing the transfer coefficients.
        """

        matrices = []  # Initialize an empty list to store matrices
        
        for distance in self.distances:  # Iterate over distances
            
            # Initialize submatrices with complex zeros
            x1_y1 = np.zeros((30, 30), dtype=np.complex128)
            x2_y1 = np.zeros((30, 30), dtype=np.complex128)
            x1_y2 = np.zeros((30, 30), dtype=np.complex128)
            x2_y2 = np.zeros((30, 30), dtype=np.complex128)
            
            # Fill diagonal elements with complex exponential terms
            for i in range(30):
                x1_y1[i][i] = np.exp(1j * self.k_m[i] * distance)
                x2_y2[i][i] = np.exp(-1j * self.k_m[i] * distance)
            
            # Construct the transfer matrix for the current distance
            p_n = np.array([
                [x1_y1, x2_y1],
                [x1_y2, x2_y2]
            ])
            matrices.append(p_n)  # Append the transfer matrix to the list

        return matrices   

    def generate_k_m(self) -> List[float]:
        """
        Generates a list containing the wavenumbers for a set of modes.

        Returns:
            List[float]: A list containing the wavenumbers for the modes.
                Each element represents the wavenumber for one mode.
        """

        k_m = []  # Initialize an empty list to store wavenumbers
        
        for i in range(1, 31):  # Iterate over mode indices from 1 to 30
            
            # Calculate the wavenumber for the current mode
            k_i = np.sqrt(30.5**2 - i**2)
            k_m.append(k_i)  # Append the wavenumber to the list
        
        return k_m  # Return the list of wavenumbers
 







class Functions:
      
    @staticmethod
    def visualize_matrix(mat):
        # Plot the absolute values of the complex matrix
        plt.imshow(np.abs(mat), cmap='viridis')
        plt.colorbar(label='Absolute Value')  # Add color bar to show the scale
        plt.show()   
    @staticmethod     
    def randint_x_chance_for_target( 
        chance: float, 
        target: int, 
        interval: Optional[Tuple[int]] = (1, 30)
    ) -> int:
        
        # Generate a random number between 1 and 100
        random_percentage = random.uniform(0, 1)
        
        # Given probability for the number to be the target number
        if random_percentage <= chance:
            
            return target
        
        else:
            # Remaining probability for other numbers (excluding target)
            while True:
                random_number = random.randint(*interval)
                if random_number != target:
                    return random_number
    @staticmethod
    def visualize_matrix_of_matrices(matrices: np.ndarray, title: str):
        num_rows, num_cols = matrices.shape[0], matrices.shape[1]  # Assuming matrices is a 2D numpy array
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 5))  # Adjust figsize as needed
        
        for i in range(num_rows):
            for j in range(num_cols):
                im = axs[i, j].imshow(np.abs(matrices[i, j]), cmap='viridis')  # Plot all matrices at position (i, j)
                #axs[i, j].set_title(f'$P_{{n_{{x_{i+1}y_{j+1} }}}}$')
                axs[0, 0].set_title("t - Magnitude")
                axs[0, 1].set_title("r' - Magnitude")
                axs[1, 0].set_title("r - Magnitude")
                axs[1, 1].set_title("t' - Magnitude")
                axs[i, j].axis('on')  # Turn off axis labels
                fig.colorbar(im, ax=axs[i, j], label="Magnitude")  # Add colorbar

        fig.suptitle(title)
        plt.tight_layout()
        plt.show()

    def visualize_matrix_of_matrices_complex(matrices, title: str):
        """
        Visualizes a matrix of matrices.

        This method plots a matrix of matrices, where each element of the input
        matrices array is a matrix itself.

        Args:
            matrices (np.ndarray): The input matrix of matrices.
            title (str): The title of the plot.
        """
        num_rows, num_cols = matrices.shape[0], matrices.shape[1]  # Assuming matrices is a 2D numpy array
        fig, axs = plt.subplots(num_rows, num_cols * 2, figsize=(20, 20))  # Double the number of columns
        
        for i in range(num_rows):
            for j in range(num_cols):
                # Plot magnitude of complex values
                im_mag = axs[i, 2*j].imshow(np.real(matrices[i, j]), cmap='viridis')
                axs[0, 2*0].set_title("t - Real Part")
                axs[0, 2*1].set_title("r' - Real Part")
                axs[1, 2*0].set_title("r - Real Part")
                axs[1, 2*1].set_title("t' - Real Part")

                
                axs[i, 2*j].axis('on')
                fig.colorbar(im_mag, ax=axs[i, 2*j], label="", shrink=0.45)
                
                # Plot imaginary part of complex values
                im_imag = axs[i, 2*j+1].imshow(np.imag(matrices[i, j]), cmap='viridis')
                axs[0, 2*0+1].set_title("t - Imaginary Part")
                axs[0, 2*1+1].set_title("r' - Imaginary Part")
                axs[1, 2*0+1].set_title("r - Imaginary Part")
                axs[1, 2*1+1].set_title("t' - Imaginary Part")
                
                axs[i, 2*j+1].axis('on')
                fig.colorbar(im_imag, ax=axs[i, 2*j+1], label="", shrink=0.45)
        
        fig.suptitle(title)  # Set the overall title of the plot
        plt.tight_layout()
        plt.show()
   
        
           
if __name__ == "__main__":
    
    homework = Homework2()