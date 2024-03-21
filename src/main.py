from matplotlib import pyplot as plt
from typing import List, Optional, Tuple, Dict
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import constants as cnst
from copy import copy
import json
import os
    
class Task1:
    
    C_L: float
    C_R: float
    C_g1: float
    C_g2: float
    C_12: float
    E_C: np.ndarray
    N_1: int
    N_2: int
    
    def __init__(self) -> None:
        
        self.gen_Cs()
        self.E_C = self.gen_E_C()
        self.task_a()
    
    def task_a(self):
        
        # Calculate q's
        
        V_L = np.zeros((200, ))
        V_R = np.zeros((200, ))
        V_g1 = np.linspace(-.1, .10, 200)
        V_g2 = np.linspace(-.10, .10, 200)      
        
        matrix = np.array([
            [self.C_L, self.C_g1, 0, 0],
            [0, 0, self.C_g2, self.C_R]
        ])
        
        vector = np.array([
            V_L, V_g1, V_g2, V_R
        ]) 
                
        q_n = np.dot(
            matrix, vector
        )  
        
        q_1, q_2 = q_n[0], q_n[1]
        
        U = 0
        
        N = [np.ones(q_1.shape), np.ones(q_2.shape)]
        N_1s = [0, 1, 2]
        N_2s = [0, 1, 2]
        
        plt.figure(figsize=(8, 6))
        
        for N_1 in N_1s:
            
            U = 0
            
            for N_2 in N_2s:
                
                for i in range(2):
                    
                    for j in range(2):
                    
                        U += (
                            self.E_C[i][j]
                            * (N[i] * N_1 - q_n[i]/cnst.elementary_charge)
                            * (N[j] * N_2 - q_n[j]/cnst.elementary_charge)
                            )
                        
                plt.plot(U, U)
        

        
        

        plt.xlabel('V_g2')
        plt.ylabel('V_g1')
        plt.title('q_1 plot')
        plt.show()

        
                
                
    
    def gen_Cs(self):
        
        self.C_L = (
            0.2 
          * np.abs(cnst.elementary_charge) 
          * 1e-3
        )
        
        self.C_R = copy(self.C_L)
        
        self.C_g1 = (
            0.3
          * np.abs(cnst.elementary_charge)
          * 1e-3
        )
        self.C_g2 = copy(self.C_g1)
        
        self.C_12 = 0
    
        self.C_1 = self.C_L + self.C_g1 + self.C_12
        self.C_2 = copy(self.C_1)
    
    def gen_E_C(self):
        
        fraction = (
            np.square(cnst.elementary_charge)
          / (
                2 * (self.C_1*self.C_2 - np.square(self.C_12))
            )
        )
        
        matrix = np.array([
            [self.C_2, self.C_12],
            [self.C_12, self.C_1]
        ])
        
        E_C = fraction * matrix
        
        return E_C
 
class Task2:
    
    alpha: float
    k_m: List[float]
    scattering_matrix: np.ndarray
    
    __slots__ = ['alpha', 'k_m', 'scattering_matrix']
    
    def __init__(self, alpha: Optional[float] = 0.035) -> None:
        """
        Initializes the Homework2 object.

        Args:
            **kwargs: Additional keyword arguments. 
                alpha (float, optional): Value of alpha parameter. Defaults to 0.035.
        """
                
        # Define alpha
        self.alpha = alpha
        
        # Generate the k_m vector
        self.k_m = self.generate_k_m()

        # Generate the scattering matrix.
        self.scattering_matrix = self.generate_scattering_matrix()
                
    def iterate_n_times__with_random_coordinates(self, iterations: int, name: str) -> Tuple[List[float]]:
        """
        Calculates the total scattering matrix and channel conductance a given amount of times, 
        each time generating random scattering points and free spaces. For each iteration the channel conductance
        is calculated, and after every iteration the conductances are plotted in a scatterplot, and the variance of
        the conductance is calculated. To avoid repeating this tedious calculation more than necessary, the 
        calculated values for the conductance is stored as a .json file.

        Args:
            name (str): Name of the json file the conductance values will be saved as.
        
        Returns:
            None
        """
        
        # If a run with the same name have been performed previously, load that one
        # instead of performing all these calculations over again.
        if os.path.exists(f"Results/saved_plots/{name}.json"):
            Resources.plot_conductances(Resources.get_saved_plot(name))
            return
        
        conductances = []
        iteration_nr = []

        # Iterate a given nr of times specified by the input parameter.
        for i in range(iterations):
            
            # Generate 600 randomly positioned x-coordinates of scattering 
            # points and the 601 distances of free space in between them.
            _, dist = self.generate_scattering_points_and_free_space()
            
            # Generate the 601 scattering matrices representing each
            # region of free space inbetween the scattering points.
            p_n = self.generate_p_n(dist)
            
            # Calculate the total scattering matrix of the channel
            total_S_matrix = self.calculate_total_scattering_matrix(p_n)
            
            # Calculate the conductance of the channel based on 
            # the total scattering matrix of the channel.
            conductance = self.calculate_conductance_from_scattering_matrix(total_S_matrix)
            
            # Append the calculated conductance and iteration nr. to 
            # their corresponding lists
            conductances.append(conductance)
            iteration_nr.append(i+1)
            
            # Calculation is long, so update the console for each iteration
            print(f"Iteration {iteration_nr[-1]} / {iterations}")

        # Convert to numpy arrays
        conductances = np.array(conductances)

        # Make a list of normalized conductance values with units e^2/h
        normalized_conductances = (
            conductances / (np.square(cnst.elementary_charge) / cnst.h)
        )
        
        # Store the calculated conductances and statistical values
        calculated_values = {
            "iterations" : iteration_nr,
            
            "conductances" : conductances.tolist(),
            "standard deviation" : np.std(conductances),
            "variance" : np.var(conductances),
            "mean" : np.mean(conductances),
            
            "normalized conductances" : normalized_conductances.tolist(),
            "normalized standard deviation" : np.std(normalized_conductances),
            "normalized variance" : np.var(normalized_conductances),
            "normalized mean" : np.mean(normalized_conductances)
        }

        # Write the calculated values to a JSON file
        with open(f"Results/saved_plots/{name}.json", "w") as json_file:
            json.dump(calculated_values, json_file)
        
        # Write calculated values to a text file
        with open(f"Results/calculated_values/{name}.txt", 'w') as f:
            for key, value in calculated_values.items():
                f.write('%s:\t%s\n' % (key, value))
        
        # Plot the conductances in a scatterplot
        Resources.plot_conductances(calculated_values)
                     
    def calculate_total_scattering_matrix(self, p_n: np.ndarray) -> np.ndarray:
        """
        Calculates the total scattering matrix through the channel
        with 600 scattering points and 601 regions of free space.

        This method calculates the total scattering matrix by performing
        a series of mathematical operations on the p_n and scattering matrices 
        as found in task 2a) from the problem set.

        Returns:
            np.ndarray: The total scattering matrix.
        """
        p_n.reverse()  # Reverse the order of p_n matrices for calculation
        
        S_tot = None
        first_calculation = True
        for p in p_n:
            
            if first_calculation:
                # If first iteration, just add the last p matrix to subcalculation.
                S_tot = p
                first_calculation = False
            else: 
                # Matrix multiply the current p matrix with the product 
                # of the previous calculations.
                S_tot = self.smack_together_two_scattering_matrices(
                    m_1 = p, m_2 = S_tot
                )
                
            # Perform the found matematical operation with a scattering matrix and the
            # product of the previous calculations.
            S_tot = self.smack_together_two_scattering_matrices(
                m_1 = self.scattering_matrix, m_2 = S_tot
            )

        # Return the total scattering matrix
        return S_tot

    def generate_scattering_matrix(self) -> np.ndarray:
        """
        Generates a scattering matrix as described in the problem set.

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
         
    def generate_p_n(self, distances: List[float]) -> List[np.ndarray]:
        """
        Generates a list of matrices for a given set of distances.

        Returns:
            List[np.ndarray]: A list of matrices, where each matrix represents
                the scattering matrix corresponding to a specific distance of free space.
                Each matrix has shape (2, 2, 30, 30) representing the scattering coefficients.
        """

        matrices = []  # Initialize an empty list to store matrices
        
        for distance in distances:  # Iterate over distances
            
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

    @staticmethod
    def calculate_conductance_from_scattering_matrix(S: np.ndarray) -> float:
        """
        Calculates the conductance from a given scattering matrix.

        Args:
            S (np.ndarray): Scattering matrix.

        Returns:
            float: Calculated conductance.
        """
        # Extract the t_tot coefficient from the total
        # scattering matrix of the channel.
        t_tot = S[0][0]
        
        # Take the conjugate transpose of transmission matrix.
        t_dagger = np.conjugate(t_tot).T
        
        # Calculate transmission matrix by matrix multiplying t with it's
        # transposed conjugate.
        T = np.matmul(t_dagger, t_tot)
        
        # Compute all eigenvalues of the transmission matrix. In theory
        # these eigenvalues are real, but python will give you complex eigenvalues
        # with very small imaginary parts, in the order of 1e-17 to 1e-20. I suspect
        # these are numerical artifacts related to rounding errors and numerical prescision
        # within python. Therefor the np.real() function is applied to discard the imaginary parts.
        T_n = np.real(np.linalg.eigvals(T))
        
        print(T_n)
        
        # Calculate the conductance, which is a product of the conductance quantum and
        # the sum of the eigenvalues of the transmission matrix T.
        conductance = (
            (np.square(cnst.elementary_charge) / cnst.h) * np.sum(T_n)
        )
        
        return conductance

    @staticmethod
    def smack_together_two_scattering_matrices(m_1: np.ndarray, m_2: np.ndarray) -> np.ndarray:
        """
        Combine two scattering matrices into a single total scattering matrix.

        Parameters:
            m_1 (np.ndarray): Scattering matrix 1.
            m_2 (np.ndarray): Scattering matrix 2.

        Returns:
            np.ndarray: Total scattering matrix.
        """    
        
        # Extract the transmission and reflection coeff. matrices of both matrices
        
        t_1 = m_1[0][0]  # Transmission coefficient t_1 matrix
        rp_1 = m_1[1][0]  # Reflection coefficients r_1' matrix
        r_1 = m_1[0][1]  # Reflection coefficients r_1 matrix
        tp_1 = m_1[1][1]  # Transmission coefficients t_1' matrix
        
        t_2 = m_2[0][0]  # Transmission coefficient t_2 matrix
        rp_2 = m_2[1][0]  # Reflection coefficients r_2' matrix
        r_2 = m_2[0][1]  # Reflection coefficients r_2 matrix
        tp_2 = m_2[1][1]  # Transmission coefficients t_2' matrix

        # Calculate the total coefficients
        t_tot = np.matmul(
            t_2,
            np.matmul(
                np.linalg.inv(
                    np.eye(30) - np.matmul(
                        rp_1, 
                        r_2
                    )
                ),
                t_1
            )
        )
        r_tot = (
            r_1 + 
            np.matmul(
                tp_1,
                np.matmul(
                    r_2,
                    np.matmul(
                        np.linalg.inv(
                            np.eye(30) - np.matmul(
                                rp_1,
                                r_2
                            )
                        ),
                        t_1
                    )
                )
            )
        )
        tp_tot = (
            np.matmul(
                tp_1,
                np.matmul(
                    r_2,
                    np.matmul(
                        np.linalg.inv(
                            np.eye(30) - np.matmul(
                                rp_1,
                                r_2
                            )
                        ),
                        np.matmul(
                            rp_1,
                            tp_2
                        )
                    )
                )
            )
            + np.matmul(
                tp_1, 
                tp_2
            )
        )
        rp_tot = (
            np.matmul(
                t_2,
                np.matmul(
                    np.linalg.inv(
                        np.eye(30) - np.matmul(
                            rp_1,
                            r_2
                        )
                    ),
                    np.matmul(
                        rp_1,
                        tp_2
                    )
                )
            )
            + rp_2
        )
        
        # Generate the total scattering matrix
        S_tot = np.array([
            [t_tot, rp_tot],
            [r_tot, tp_tot]
        ])
        
        return S_tot
 
    @staticmethod
    def generate_scattering_points_and_free_space() -> Tuple[List[float], List[float]]:
        """
        Generates random coordinates for scattering points and distances of 
        free space between them.

        Returns:
            Tuple[List[float], List[float]]: A tuple containing two lists.
                The first list contains randomly generated x-coordinates.
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
        
        nr_coordinates = len(coordinates)
        # Calculate distances between consecutive coordinates
        for index in range(0, nr_coordinates):
            
            if index == 0:
                # Append the distance between x=0 and 
                # the first scattering point.
                distance = coordinates[0]
                distances.append(distance)
            else:    

                # Calculate the distance between i-th scattering point
                # and the previous scattering point.
                distance = coordinates[index] - coordinates[index - 1]
                distances.append(distance)
                
                # If the i-th scattering point is the last, calculate
                # the distance from it to the last x-coordinate 60.000.
                if index == nr_coordinates - 1:
                    distance = 60000 - coordinates[i]
                    distances.append(distance)

        distances = np.array(distances)
        
        return coordinates, distances  # Return the generated coordinates and distances

    @staticmethod
    def generate_k_m() -> List[float]:
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

class Resources:
    
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
            'Conductance [$\\frac{e^2}{h}$]' 
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
        #plt.ylim(29.5, 30.5)
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
        plt.imshow(np.abs(mat), cmap='viridis')
        plt.colorbar(label='Absolute Value')
        plt.show()

    @staticmethod
    def randint_x_chance_for_target(chance: float, target: int, interval: Optional[Tuple[int]] = (1, 30)) -> int:
        """
        Generate a random integer with a certain chance of being the target.

        This method generates a random integer within the specified interval
        with a given probability for it to be the target number.

        Args:
            chance (float): The probability of generating the target number.
            target (int): The target number.
            interval (Tuple[int], optional): The interval for generating random numbers. Defaults to (1, 30).

        Returns:
            int: The randomly generated integer.
        """
        random_percentage = random.uniform(0, 1)
        if random_percentage <= chance:
            return target
        else:
            while True:
                random_number = random.randint(*interval)
                if random_number != target:
                    return random_number

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
        
   
if __name__ == "__main__":
    
    # Initialize with alpha = 0.035
    #alpha_0_035 = Task2(alpha = 0.035)
    
    # Do a run of 200 iterations with alpha = 0.035
    #alpha_0_035.iterate_n_times__with_random_coordinates(
    #    iterations = 200, 
    #    name = "200_alpha_0_035"
    #)
    
    # Initialize with alpha = 0.035
    #alpha_0 = Task2(alpha = 0)
    
    # Do a run of 200 iteration with alpha = 0
    #alpha_0.iterate_n_times__with_random_coordinates(
    #    iterations = 3, 
    #    name = "200_alpha_0"
    #)
    Resources.plot_conductances(Resources.get_saved_plot("200_alpha_0_035"), normalized=False)
    

