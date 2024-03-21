from matplotlib import pyplot as plt
from typing import List, Optional, Tuple
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import constants as cnst
from copy import copy

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
    
    def __init__(self, alpha: Optional[float] = 0.035) -> None:
        """
        Initializes the Homework2 object.

        Args:
            **kwargs: Additional keyword arguments. 
                alpha (float, optional): Value of alpha parameter. Defaults to 0.035.
        """
        
        # Generate a list with x-coordinates of 600 scattering points,
        # and a list with 601 distances between scattering points.
        self.scattering_points, self.distances = self.generate_scattering_points_and_free_space()
        
        # Define alpha
        self.alpha = alpha
        
        # Generate the k_m vector
        self.k_m = self.generate_k_m()
        
        # Generate the 601 p_n matrices
        #self.p_n = self.generate_p_n(self.distances)

        # Generate the scattering matrix.
        self.scattering_matrix = self.generate_scattering_matrix()

        #self.total_scattering_matrix = self.calculate_total_scattering_matrix(self.p_n)
        
        #self.calculate_conductance_from_scattering_matrix(self.total_scattering_matrix)
        self.iterate_200_times_random_coordinates()
        
    @staticmethod
    def calculate_conductance_from_scattering_matrix(S: np.ndarray) -> float:
        
        t = S[1][1]
        t_dagger = np.conjugate(t).T
        
        T = np.matmul(t_dagger, t)
        T_n = np.real(np.linalg.eigvals(T))
        
        conductance = np.sum(T_n)

        print(conductance)

        return conductance
    
    def iterate_200_times_random_coordinates(self):
        
        conductances = []
        iterations = []
        
        for i in range(50):
            s_points, dist = self.generate_scattering_points_and_free_space()
            p_n = self.generate_p_n(dist)
            total_S_matrix = self.calculate_total_scattering_matrix(p_n)
            conductance = self.calculate_conductance_from_scattering_matrix(total_S_matrix)
            conductances.append(conductance)
            iterations.append(i+1)
        
        # Plot conductances versus iterations
        plt.scatter(iterations, conductances, marker='o', linestyle='-')
        plt.xlabel('Iterations')
        plt.ylabel('Conductance')
        plt.title('Conductance vs Iterations')
        plt.grid(True)
        plt.show()
              
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
        
    def calculate_total_scattering_matrix(self, p_n) -> np.ndarray:
        """
        Calculates the total scattering matrix through the channel.

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
                
            # Perform matrix multiplication with the transfer matrix and the
            # product of the previous calculations.
            S_tot = self.smack_together_two_scattering_matrices(
                m_1 = self.scattering_matrix, m_2 = S_tot
            )

        # Return the total scattering matrix
        return S_tot

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
    
    def generate_p_n(self, distances: List[float]) -> List[np.ndarray]:
        """
        Generates a list of matrices for a given set of distances.

        Returns:
            List[np.ndarray]: A list of matrices, where each matrix represents
                the transfer matrix corresponding to a specific distance.
                Each matrix has shape (2, 2, 30, 30) representing the transfer coefficients.
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

class Functions:
    
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
    
if __name__ == "__main__":
     
    #task1 = Task1()
    homework_alpha_0035 = Homework2(alpha = 0.035)
    #Functions.visualize_matrix_of_matrices(homework_alpha_0035.total_scattering_matrix, f"$\\hat{{S}}_{{tot}}$ (magnitude) for  $\\alpha = 0.000035$ ")
    #Functions.visualize_matrix_of_matrices_complex(homework_alpha_0035.total_scattering_matrix, f"$\\hat{{S}}_{{tot}}$ (real and imaginary parts) for  $\\alpha = 0.000035$ ")
    #homework_alpha_0 = Homework2(alpha = 0)
    #Functions.visualize_matrix_of_matrices(homework_alpha_0.total_scattering_matrix, f"$\\hat{{S}}_{{tot}}$ (magnitude) for $\\alpha = 0$ ")
    #Functions.visualize_matrix_of_matrices_complex(homework_alpha_0.total_scattering_matrix, f"$\\hat{{S}}_{{tot}}$ (real and imaginary parts) for  $\\alpha = 0$ ")