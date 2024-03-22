import numpy as np
import os
import json
import random
import scipy.constants as cnst
from resources import Resources
from typing import List, Optional, Tuple


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
         
           
    def iterate_n_times_with_random_coordinates(self, iterations: int, name: str) -> Tuple[List[float]]:
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

        # Make a list of normalized conductance values with units h/e^2
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

    def calculate_transmission_probability(self) -> float:
        
        # Generate 600 randomly positioned x-coordinates of scattering 
        # points and the 601 distances of free space in between them.
        _, dist = self.generate_scattering_points_and_free_space()
            
        # Generate the 601 scattering matrices representing each
        # region of free space inbetween the scattering points.
        p_n = self.generate_p_n(dist)
            
        # Calculate the total scattering matrix of the channel
        total_S_matrix = self.calculate_total_scattering_matrix(p_n)
        
        # Extract the t_tot coefficient from the total
        # scattering matrix of the channel.
        t_tot = total_S_matrix[0][0]
        
        # Take the conjugate transpose of t_tot matrix.
        t_dagger = np.conjugate(t_tot).T
        
        # Calculate transmission matrix by matrix multiplying t with it's
        # transposed conjugate.
        T = np.matmul(t_dagger, t_tot)
        
        Resources.visualize_matrix(T)
        
        # Compute all eigenvalues of the transmission matrix. In theory
        # these eigenvalues are real, but python will give you complex eigenvalues
        # with very small imaginary parts, in the order of 1e-17 to 1e-20. I suspect
        # these are numerical artifacts related to rounding errors and numerical prescision
        # within python. Therefore the np.real() function is applied to discard the imaginary parts.
        T_n = np.real(np.linalg.eigvals(T))
        
        return T_n
         
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
