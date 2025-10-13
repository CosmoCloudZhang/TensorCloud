import numpy
import scipy
from itertools import product

def element1(chi1, chi2, power1, power2):

    a = 1 - chi1 / chi2
    y = 1 - power1 / power2

    if a == 1:

        formula = 1 / 12

    else: 

        formula =((a * (2 * ( - 2 + a) * a + 6 * y - a * (3 + a) * y)) / (2 * ( - 1 + a)) + (2 * a - 3 * y) * numpy.log(1 - a)) / a**3
        print(formula)
        print("/n")

    coefficient = power2 *formula /chi2
    
    return coefficient

def element2(chi1, chi2, power1, power2):

    a = 1 - chi1 / chi2
    y = 1 - power1 / power2

    if a == 1:

        formula = 1 / 12

    else: 

        formula = (( - a) * ( - 6 * y + a * (4 + y)) + 2 * (a**2 + 3 * y - 2 * a * (1 + y)) * numpy.log(1 - a)) / (2 * a**3)

    coefficient = power2 *formula /chi2
    
    return coefficient

def element3(chi1, chi2, power1, power2):

    a = 1 - chi1 / chi2
    y = 1 - power1 / power2

    if a == 1:

        formula = 1 / 4

    else: 

        formula = ((1 / 2) * a * ( - 6 * y + a * (4 - 2 * a + 5 * y)) - ( - 1 + a) * ( - 3 * y + a * (2 + y)) * numpy.log(1 - a)) / a**3

    coefficient = power2 *formula /chi2
    
    return coefficient

def element(n, i , j, chi_grid, power_grid, redshift_grid):

    ell_size = power_grid.shape[0] - 1
    grid_size = chi_grid.shape[0] - 1

    if (i < n < grid_size) | (j < n < grid_size):
        
        elements = numpy.zeros(ell_size + 1)

    elif (n == i < grid_size) & (n == j < grid_size):

        elements = element1(chi_grid[n], chi_grid[n + 1], power_grid[:,n], power_grid[:,n + 1])

    elif ((n == i < grid_size) & (n + 1 == j < grid_size)) | ((n + 1 == i < grid_size) & (n == j < grid_size)) :

        elements = element2(chi_grid[n], chi_grid[n + 1], power_grid[:,n], power_grid[:,n + 1])

    elif (n + 1 == i < grid_size) & (n + 1 == j < grid_size):
        
        elements = element3(chi_grid[n], chi_grid[n + 1], power_grid[:,n], power_grid[:,n + 1])

    else: 

        elements = numpy.zeros(ell_size + 1)

    return elements

def coefficient(chi_grid, power_grid, redshift_grid):
    
    grid_size = chi_grid.shape[0] - 1
    ell_size = power_grid.shape[0] - 1
    coefficients = numpy.zeros((grid_size + 1, grid_size + 1, ell_size + 1))
    
    for n in range(grid_size):
        
        for i in range(n, grid_size + 1):
            
            for j in range(n, grid_size + 1):
                
                coefficients[i, j, :] = coefficients[i, j, :] + element(n, i, j, chi_grid, power_grid, redshift_grid)
                
    return coefficients

def function(amplitude, phi_grid1, phi_grid2, chi_grid, power_grid, redshift_grid):

    bin_size1 = phi_grid1.shape[0]
    bin_size2 = phi_grid2.shape[0]
    grid_size = chi_grid.shape[0] - 1
    ell_size = power_grid.shape[0] - 1
    functions = numpy.zeros((bin_size1, bin_size2, ell_size + 1))
    coefficients = coefficient(chi_grid, power_grid, redshift_grid)
    
    for m1 in range(bin_size1):
        
        for m2 in range(bin_size2):
            
            for i in range(grid_size + 1):

                for j in range(grid_size + 1):
                    
                    functions[m1, m2, :] = functions[m1, m2, :] + amplitude * coefficients[i, j, :] * phi_grid1[m1, i] * phi_grid2[m2, j]

    return functions