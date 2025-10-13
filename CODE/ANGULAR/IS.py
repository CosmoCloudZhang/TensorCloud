import numpy
import scipy
from itertools import product

def element1(chi1, chi2, power1, power2, redshift1, redshift2, growth1, growth2):

    a = 1 - chi1 / chi2
    y = 1 - power1 / power2
    z = 1 - (1 + redshift1) / (1 + redshift2)
    b = (1/growth2-1/growth1)*chi2/(chi2-chi1)
    c = (chi2/growth1-chi1/growth2)/(chi2-chi1)

    if a == 1:

        formula = (b * (154 - 89 * z) + 7 * c * (63 - 41 * z)) / 176400

    else: 

        formula =- ((1 / (3600 * a**4)) * (a * (50 * a**2 * ((12 + a * (6 + a * ( - 32 + 9 * a))) * b - 12 * ( - 6 + ( - 3 + a) * a) * c) - 
5 * a * ((60 + a * (30 + a * (20 + 9 * a * ( - 25 + 8 * a)))) * b + 10 * (60 + a * (30 + (20 - 9 * a) * a)) * c) * y + 
( - 5 * a * (60 + a * (30 + a * (20 + 9 * a * ( - 25 + 8 * a)))) * b + 50 * a * ( - 60 + a * ( - 30 + a * ( - 20 + 9 * a))) * c + 
3 * ((60 + a * (30 + a * (20 + a * (15 + 4 * a * ( - 72 + 25 * a))))) * b + 15 * (60 + a * (30 + a * (20 + (15 - 8 * a) * a))) * c) * y) * z) + 
60 * (10 * a**2 * (b + 6 * c) + 3 * (b + 15 * c) * y * z - 5 * a * (b + 10 * c) * (y + z) - 5 * a**4 * (b + c) * (6 - 4 * y - 4 * z + 3 * y * z) + 
a**5 * b * (20 - 15 * z + 3 * y * ( - 5 + 4 * z))) * numpy.log(1 - a)))

    coefficient = chi2* power2 * (1 + redshift2) * formula
    
    return coefficient

def element2(chi1, chi2, power1, power2, redshift1, redshift2, growth1, growth2):

    a = 1 - chi1 / chi2
    y = 1 - power1 / power2
    z = 1 - (1 + redshift1) / (1 + redshift2)
    b = (1/growth2-1/growth1)*chi2/(chi2-chi1)
    c = (chi2/growth1-chi1/growth2)/(chi2-chi1)
    if a == 1:

        formula = (b * (140 - 65 * z) + 14 * c * (21 - 11 * z)) / 176400

    else:

        formula =  - ((1 / (3600 * a**4)) * (a * ( - 180 * (b + 15 * c) * y * z + 5 * a**3 * (300 * b + 720 * c - 40 * (b + 4 * c) * y - 40 * (b + 4 * c) * z + (11 * b + 65 * c) * y * z) + 
30 * a * (10 * (b + 10 * c) * y + b * (10 + 7 * y) * z + 5 * c * (20 + 11 * y) * z) + 30 * a**5 * b * (5 - 3 * z + y * ( - 3 + 2 * z)) + 
30 * a**2 * ( - 5 * b * (4 + 3 * y) + 3 * b * ( - 5 + y) * z + 10 * c * ( - 12 - 7 * y - 7 * z + 2 * y * z)) + 
a**4 * (b * ( - 1100 + 475 * y + 475 * z - 261 * y * z) - 30 * c * (10 - 5 * y - 5 * z + 3 * y * z))) + 
60 * (30 * a**3 * (b + 3 * c) - 5 * a**4 * (b + c) * (6 + y * ( - 2 + z) - 2 * z) - 3 * (b + 15 * c) * y * z - 10 * a**2 * (b + 6 * c) * (1 + y + z) + 
5 * a * (b + 10 * c) * (y + z + y * z) + a**5 * b * (10 - 5 * y - 5 * z + 3 * y * z)) * numpy.log(1 - a)))

    coefficient = chi2* power2 * (1 + redshift2) * formula
    
    return coefficient

def element3(chi1, chi2, chi3, chi4, chi5, power1, power2, redshift1, redshift2, growth1, growth2):

    a = 1 - chi1 / chi2
    b = (chi5-chi3)/(2*chi2)
    c = chi3*numpy.log(chi4/chi3)/(chi4-chi3)-chi5*numpy.log(chi5/chi4)/(chi5-chi4)
    y = 1 - power1 / power2
    z = 1 - (1 + redshift1) / (1 + redshift2)
    d = (1/growth2-1/growth1)*chi2/(chi2-chi1)
    e = (chi2/growth1-chi1/growth2)/(chi2-chi1)

    if a == 1:

        formula =(1 / 420) * (c * (d * (14 - 4 * z) - 7 * e * ( - 3 + z)) - 7 * b * (d * ( - 3 + z) + e * ( - 5 + 2 * z)))

    else:

        formula = (1 / (60 * a**3)) * (a * ( - 60 * b * e * y * z + a**4 * c * d * ( - 20 + 15 * y + 15 * z - 12 * y * z) + 10 * a**2 * b * e * ( - 6 + 3 * y + 3 * z - 2 * y * z) + 30 * a * b * e * (2 * y + 2 * z - y * z) + 
5 * a**3 * ((b + c) * d + c * e) * (6 - 4 * y - 4 * z + 3 * y * z)) - 60 * b * e * (a - y) * (a - z) * numpy.log(1 - a))
    

    coefficient = chi2* power2 * (1 + redshift2) * formula

    return coefficient

def element4(chi1, chi2, chi3, chi4, chi5, power1, power2, redshift1, redshift2, growth1, growth2):

    a = 1 - chi1 / chi2
    b = (chi5-chi3)/(2*chi2)
    c = chi3*numpy.log(chi4/chi3)/(chi4-chi3)-chi5*numpy.log(chi5/chi4)/(chi5-chi4)
    y = 1 - power1 / power2
    z = 1 - (1 + redshift1) / (1 + redshift2)
    d = (1/growth2-1/growth1)*chi2/(chi2-chi1)
    e = (chi2/growth1-chi1/growth2)/(chi2-chi1)

    if a == 1:

        formula = (1 / 420) * (2 * c * ( - 5 * d * ( - 7 + z) - 7 * e * ( - 6 + z)) - 7 * b * (2 * d * ( - 6 + z) + 3 * e * ( - 5 + z)))

    else:

        formula = (1 / (60 * a**3)) * (a * (5 * a**3 * ((b + c) * d + c * e) * (6 + y * ( - 2 + z) - 2 * z) + 60 * b * e * y * z + a**4 * c * d * ( - 10 + 5 * y + 5 * z - 3 * y * z) + 
10 * a**2 * b * e * (6 + 3 * y + 3 * z - y * z) - 30 * a * b * e * (2 * y + (2 + y) * z)) - 60 * ( - 1 + a) * b * e * (a - y) * (a - z) * numpy.log(1 - a))
         

    coefficient = chi2* power2 * (1 + redshift2) * formula

    return coefficient

def element5(chi1, chi2, chi3, power1, power2, redshift1, redshift2, growth1, growth2):

    a = 1 - chi1 / chi2
    b = chi3 / chi2 - 1
    y = 1 - power1 / power2
    z = 1 - (1 + redshift1) / (1 + redshift2)
    c = (1/growth2-1/growth1)*chi2/(chi2-chi1)
    d = (chi2/growth1-chi1/growth2)/(chi2-chi1)

    if a == 1:

        formula =  (b * (14 * (7 + 5 * b) * d + c * (62 - 14 * b * ( - 3 + z) - 19 * z) - 4 * (9 + 7 * b) * d * z) + 4 * (1 + b) * (7 * d * ( - 3 + z) + 2 * c * ( - 7 + 2 * z)) * numpy.log(1 + b)) / (1680 * b)

    else:

        formula = (1 / (3600 * a**4)) * ( - 60 * (10 * a**3 * (c + 3 * (3 + b) * d) + 5 * a * (c + 10 * d) * y - 3 * (c + 15 * d) * y * z + a * (5 * c + 50 * d + 3 * (c + 5 * (5 + 2 * b) * d) * y) * z - 
5 * a**2 * (12 * d + 2 * (8 + 3 * b) * d * (y + z) + c * (2 + y + z)) + 5 * a**4 * (c + d) * (6 - 4 * z + y * ( - 4 + 3 * z)) + a**6 * c * (20 - 15 * z + 3 * y * ( - 5 + 4 * z)) + 
a**5 * (c * ( - 50 + 35 * y + 35 * z - 27 * y * z) - 5 * d * (6 - 4 * y - 4 * z + 3 * y * z))) * numpy.log(1 - a) + 
a * (5 * a * (10 * a * ((12 + a * ( - 6 + a * ( - 2 + 17 * a + 18 * b))) * c + 72 * d - 6 * a * (5 * a + 6 * (2 + b)) * d) + 
(( - 60 + a * (30 + a * (10 + a * (5 - 117 * a - 120 * b)))) * c + 10 * ( - 60 + a * (66 + 28 * a + 17 * a**2 + 18 * (2 + a) * b)) * d) * y) + 
(5 * a * ( - 60 + a * (30 + a * (10 + a * (5 - 117 * a - 120 * b)))) * c + 50 * a * ( - 60 + a * (66 + 28 * a + 17 * a**2 + 18 * (2 + a) * b)) * d + 
3 * (60 * c + a * ( - 30 + a * ( - 10 + a * ( - 5 + a * ( - 3 + 148 * a + 150 * b)))) * c + 900 * d - 5 * a * (30 * (7 + 4 * b) + a * (90 + 60 * b + a * (55 + 39 * a + 40 * b))) * d) * 
y) * z + (60 * a**4 * (1 + b) * ( - 5 * c * (6 - 4 * a - 4 * y + 3 * a * y) + c * (20 - 15 * y + 3 * a * ( - 5 + 4 * y)) * z - 5 * d * (6 - 4 * y - 4 * z + 3 * y * z)) * numpy.log(1 + b)) / b))

    coefficient = chi2* power2 * (1 + redshift2) * formula
    
    return coefficient

def element6(chi1, chi2, chi3, power1, power2, redshift1, redshift2, growth1, growth2):

    a = 1 - chi1 / chi2
    b = chi3 / chi2 - 1
    y = 1 - power1 / power2
    z = 1 - (1 + redshift1) / (1 + redshift2)
    c = (1/growth2-1/growth1)*chi2/(chi2-chi1)
    d = (chi2/growth1-chi1/growth2)/(chi2-chi1)

    if a == 1:

        formula =  (b * (d * (350 - 42 * b * ( - 5 + z) - 62 * z) + c * (288 - 28 * b * ( - 6 + z) - 43 * z)) + 8 * (1 + b) * (5 * c * ( - 7 + z) + 7 * d * ( - 6 + z)) * numpy.log(1 + b)) / (1680 * b)
        
    else:

        formula = (1 / (3600 * a**4 * b)) * (a * b * ( - 180 * (c + 15 * d) * y * z + 30 * a * (10 * (c + 10 * d) * y + c * (10 + 13 * y) * z + 5 * d * (20 + (41 + 12 * b) * y) * z) + 
a**4 * (25 * c * ( - 32 - 12 * b * ( - 3 + y) + 3 * y) + c * (75 + 150 * b * ( - 2 + y) - 16 * y) * z + 5 * d * ( - 600 + y * (130 - 53 * z) + 130 * z)) + 
a**5 * c * (650 - 265 * z + y * ( - 265 + 141 * z)) + 5 * a**3 * (c * (420 + y * (50 - 7 * z) + 50 * z) + 
5 * d * (360 + 88 * y + 88 * z - 23 * y * z + 12 * b * (6 + 3 * y + 3 * z - y * z))) - 
30 * a**2 * (c * (20 + 25 * y + 25 * z + 4 * y * z) + 5 * d * (24 + 46 * z + 12 * b * z + y * (46 + 13 * z + 6 * b * (2 + z))))) - 
60 * ( - 1 + a) * b * (30 * a**3 * (c + (4 + b) * d) - 5 * a**4 * (c + d) * (6 + y * ( - 2 + z) - 2 * z) - 3 * (c + 15 * d) * y * z + a**5 * c * (10 - 5 * y - 5 * z + 3 * y * z) - 
10 * a**2 * (6 * d + 3 * (3 + b) * d * (y + z) + c * (1 + y + z)) + 5 * a * (c * (y + z + y * z) + 2 * d * (5 * y + (5 + 8 * y + 3 * b * y) * z))) * numpy.log(1 - a) + 
60 * a**5 * (1 + b) * ( - 5 * d * (6 + y * ( - 2 + z) - 2 * z) + c * ( - 5 * y * z + 10 * ( - 3 + y + z) + a * (10 - 5 * y - 5 * z + 3 * y * z))) * numpy.log(1 + b))

    coefficient = chi2* power2 * (1 + redshift2) * formula

    return coefficient

def element7(chi1, chi2, chi3, chi4, power1, power2, redshift1, redshift2, growth1, growth2):

    a = 1 - chi1 / chi2
    b = (chi4-chi3)/(2*chi2)
    c = chi3*numpy.log(chi4/chi3)/(chi4-chi3)-1
    y = 1 - power1 / power2
    z = 1 - (1 + redshift1) / (1 + redshift2)
    d = (1/growth2-1/growth1)*chi2/(chi2-chi1)
    e = (chi2/growth1-chi1/growth2)/(chi2-chi1)

    if a == 1:

        formula = (1 / 420) * (c * (d * (14 - 4 * z) - 7 * e * ( - 3 + z)) - 7 * b * (d * ( - 3 + z) + e * ( - 5 + 2 * z)))

    else: 

        formula = (1 / (60 * a**3)) * (a * ( - 60 * b * e * y * z + a**4 * c * d * ( - 20 + 15 * y + 15 * z - 12 * y * z) + 10 * a**2 * b * e * ( - 6 + 3 * y + 3 * z - 2 * y * z) + 30 * a * b * e * (2 * y + 2 * z - y * z) + 
5 * a**3 * ((b + c) * d + c * e) * (6 - 4 * y - 4 * z + 3 * y * z)) - 60 * b * e * (a - y) * (a - z) * numpy.log(1 - a))

    coefficient = chi2* power2 * (1 + redshift2) * formula

    return coefficient

def element8(chi1, chi2, chi3, chi4, power1, power2, redshift1, redshift2, growth1, growth2):

    a = 1 - chi1 / chi2
    b = (chi4-chi3)/(2*chi2)
    c = chi3*numpy.log(chi4/chi3)/(chi4-chi3)-1
    y = 1 - power1 / power2
    z = 1 - (1 + redshift1) / (1 + redshift2)
    d = (1/growth2-1/growth1)*chi2/(chi2-chi1)
    e = (chi2/growth1-chi1/growth2)/(chi2-chi1)

    if a == 1:

        formula =(1 / 420) * (2 * c * ( - 5 * d * ( - 7 + z) - 7 * e * ( - 6 + z)) - 7 * b * (2 * d * ( - 6 + z) + 3 * e * ( - 5 + z)))
        
    else:

        formula =(1 / (60 * a**3)) * (a * (5 * a**3 * ((b + c) * d + c * e) * (6 + y * ( - 2 + z) - 2 * z) + 60 * b * e * y * z + a**4 * c * d * ( - 10 + 5 * y + 5 * z - 3 * y * z) + 
10 * a**2 * b * e * (6 + 3 * y + 3 * z - y * z) - 30 * a * b * e * (2 * y + (2 + y) * z)) - 60 * ( - 1 + a) * b * e * (a - y) * (a - z) * numpy.log(1 - a))

    coefficient = chi2* power2 * (1 + redshift2) * formula

    return coefficient



def element(n, i , j, chi_grid, power_grid, redshift_grid, growth_grid):

    ell_size = power_grid.shape[0] - 1
    grid_size = chi_grid.shape[0] - 1

    if (i < n < grid_size) | (j < n < grid_size):
        
        elements = numpy.zeros(ell_size + 1)

    elif (n == j < grid_size) & (n == i < grid_size):

        elements = element1(chi_grid[n], chi_grid[n + 1], power_grid[:,n], power_grid[:,n + 1], redshift_grid[n], redshift_grid[n + 1], growth_grid[n], growth_grid[n + 1])

    elif (n == j < grid_size) & (n + 1 == i < grid_size) :

        elements = element2(chi_grid[n], chi_grid[n + 1], power_grid[:,n], power_grid[:,n + 1], redshift_grid[n], redshift_grid[n + 1], growth_grid[n], growth_grid[n + 1])

    elif (n + 1 < j < grid_size) & (n == i < grid_size):
        
        elements = element3(chi_grid[n], chi_grid[n + 1], chi_grid[j - 1], chi_grid[j], chi_grid[j + 1], power_grid[:,n], power_grid[:,n + 1], redshift_grid[n], redshift_grid[n + 1], growth_grid[n], growth_grid[n + 1])
       
    elif (n + 1 < j < grid_size) & (n + 1 == i < grid_size):

        elements = element4(chi_grid[n], chi_grid[n + 1], chi_grid[j - 1], chi_grid[j], chi_grid[j + 1], power_grid[:,n], power_grid[:,n + 1], redshift_grid[n], redshift_grid[n + 1], growth_grid[n], growth_grid[n + 1])

    elif (n + 1 == j < grid_size) & (n == i < grid_size):

        elements = element5(chi_grid[n], chi_grid[n + 1], chi_grid[n + 2], power_grid[:,n], power_grid[:,n + 1], redshift_grid[n], redshift_grid[n + 1], growth_grid[n], growth_grid[n + 1])

    elif (n + 1 == j < grid_size) & (n + 1 == i < grid_size):

        elements = element6(chi_grid[n], chi_grid[n + 1], chi_grid[n + 2], power_grid[:,n], power_grid[:,n + 1], redshift_grid[n], redshift_grid[n + 1], growth_grid[n], growth_grid[n + 1])
        
    elif (j == grid_size) & (n == i < grid_size):
        
        elements = element7(chi_grid[n], chi_grid[n + 1], chi_grid[grid_size - 1], chi_grid[grid_size], power_grid[:,n], power_grid[:,n + 1], redshift_grid[n], redshift_grid[n + 1], growth_grid[n], growth_grid[n + 1])
       
    elif (j == grid_size) & (n + 1 == i < grid_size):

        elements = element8(chi_grid[n], chi_grid[n + 1], chi_grid[grid_size - 1], chi_grid[grid_size], power_grid[:,n], power_grid[:,n + 1], redshift_grid[n], redshift_grid[n + 1], growth_grid[n], growth_grid[n + 1])

    else: 

        elements = numpy.zeros(ell_size + 1)

    return elements

def coefficient(chi_grid, power_grid, redshift_grid, growth_grid):
    
    grid_size = chi_grid.shape[0] - 1
    ell_size = power_grid.shape[0] - 1
    coefficients = numpy.zeros((grid_size + 1, grid_size + 1, ell_size + 1))
    
    for n in range(grid_size):
        
        for i in range(n, grid_size + 1):
            
            for j in range(n, grid_size + 1):
                
                coefficients[i, j, :] = coefficients[i, j, :] + element(n, i, j, chi_grid, power_grid, redshift_grid, growth_grid)
                
    return coefficients

def function(amplitude, phi_grid, chi_grid, power_grid, redshift_grid, growth_grid):

    bin_size = phi_grid.shape[0]
    grid_size = chi_grid.shape[0] - 1
    ell_size = power_grid.shape[0] - 1
    functions = numpy.zeros((bin_size, bin_size, ell_size + 1))
    coefficients = coefficient(chi_grid, power_grid, redshift_grid, growth_grid)
    
    for m1 in range(bin_size):
        
        for m2 in range(bin_size):
            
            for i in range(grid_size + 1):

                for j in range(grid_size + 1):
                    
                    functions[m1, m2, :] = functions[m1, m2, :] + amplitude * coefficients[i, j, :] * phi_grid[m1, i] * phi_grid[m2, j]

    return functions