import numpy
import scipy
from itertools import product

def element1(chi1, chi2, power1, power2, growth1, growth2):

    a = 1 - chi1 / chi2
    y = 1 - power1 / power2
    b = (1/growth2-1/growth1)*chi2/(chi2-chi1)
    c = (chi2/growth1-chi1/growth2)/(chi2-chi1)

    if a == 1:

        formula =  (1 / 60) * (2 * b + 5 * c)


    else: 

        formula = ((a * ( - 3 * a * ( - 2 + a + a**2) * b + 6 * ( - 2 + a) * a * c + ( - 6 + a * (3 + a + 2 * a**2)) * b * y - 3 * ( - 6 + a * (3 + a)) * c * y)) / (6 * ( - 1 + a)) + 
(( - a) * b + 2 * a * c + b * y - 3 * c * y) * numpy.log(1 - a)) / a**3


    coefficient = power2 *formula /chi2
    
    return coefficient

def element2(chi1, chi2, power1, power2, growth1, growth2):

    a = 1 - chi1 / chi2
    y = 1 - power1 / power2
    b = (1/growth2-1/growth1)*chi2/(chi2-chi1)
    c = (chi2/growth1-chi1/growth2)/(chi2-chi1)
    if a == 1:

        formula =  (1 / 60) * (3 * b + 5 * c)

    else: 

        formula =  (a * (a**2 * b * ( - 3 + y) - 6 * (b - 3 * c) * y + 3 * a * b * (2 + y) - 3 * a * c * (4 + y)) - 6 * (a**2 * (b - c) + (b - 3 * c) * y - a * (b - 2 * c) * (1 + y)) * numpy.log(1 - a)) / (6 * a**3)

    coefficient = power2 *formula /chi2
    
    return coefficient

def element3(chi1, chi2, power1, power2, growth1, growth2):

    a = 1 - chi1 / chi2
    y = 1 - power1 / power2
    b = (1/growth2-1/growth1)*chi2/(chi2-chi1)
    c = (chi2/growth1-chi1/growth2)/(chi2-chi1)
    if a == 1:

        formula = b / 5 + c / 4

    else: 

        formula =  ((1 / 6) * a * (3 * a * ( - 2 + 3 * a) * b - 6 * ( - 2 + a) * a * c + (6 + a * ( - 9 + 2 * a)) * b * y + 3 * ( - 6 + 5 * a) * c * y) - 
( - 1 + a) * (a**2 * b + (b - 3 * c) * y - a * b * (1 + y) + a * c * (2 + y)) * numpy.log(1 - a)) / a**3

    coefficient = power2 *formula /chi2
    
    return coefficient

def element(n, i , j, chi_grid, power_grid, growth_grid):

    ell_size = power_grid.shape[0] - 1
    grid_size = chi_grid.shape[0] - 1

    if (i < n < grid_size) | (j < n < grid_size):
        
        elements = numpy.zeros(ell_size + 1)

    elif (n == i < grid_size) & (n == j < grid_size):

        elements = element1(chi_grid[n], chi_grid[n + 1], power_grid[:,n], power_grid[:,n + 1], growth_grid[n], growth_grid[n + 1])

    elif ((n == i < grid_size) & (n + 1 == j < grid_size)) | ((n + 1 == i < grid_size) & (n == j < grid_size)) :

        elements = element2(chi_grid[n], chi_grid[n + 1], power_grid[:,n], power_grid[:,n + 1], growth_grid[n], growth_grid[n + 1])

    elif (n + 1 == i < grid_size) & (n + 1 == j < grid_size):
        
        elements = element3(chi_grid[n], chi_grid[n + 1], power_grid[:,n], power_grid[:,n + 1], growth_grid[n], growth_grid[n + 1])

    else: 

        elements = numpy.zeros(ell_size + 1)

    return elements

def coefficient(chi_grid, power_grid, growth_grid):
    
    grid_size = chi_grid.shape[0] - 1
    ell_size = power_grid.shape[0] - 1
    coefficients = numpy.zeros((grid_size + 1, grid_size + 1, ell_size + 1))
    
    for n in range(grid_size):
        
        for i in range(n, grid_size + 1):
            
            for j in range(n, grid_size + 1):
                
                coefficients[i, j, :] = coefficients[i, j, :] + element(n, i, j, chi_grid, power_grid, growth_grid)
                
    return coefficients

def function(amplitude, phi_grid, chi_grid, power_grid, growth_grid):

    bin_size = phi_grid.shape[0]
    grid_size = chi_grid.shape[0] - 1
    ell_size = power_grid.shape[0] - 1
    functions = numpy.zeros((bin_size, bin_size, ell_size + 1))
    coefficients = coefficient(chi_grid, power_grid, growth_grid)
    
    for m1 in range(bin_size):
        
        for m2 in range(bin_size):
            
            for i in range(grid_size + 1):

                for j in range(grid_size + 1):
                    
                    functions[m1, m2, :] = functions[m1, m2, :] + amplitude * coefficients[i, j, :] * phi_grid[m1, i] * phi_grid[m2, j]

    return functions