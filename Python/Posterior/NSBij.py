import numpy
import scipy
from itertools import product

def element1(chi1, chi2, power1, power2, redshift1, redshift2):

    a = 1 - chi1 / chi2
    y = 1 - power1 / power2
    z = 1 - (1 + redshift1) / (1 + redshift2)

    if a == 1:

        formula = (63 - 41 * z) / 25200

    else: 

        formula =(1 / (720 * a**4)) * (a * (10 * a * (12 * a * ( - 6 + ( - 3 + a) * a) + (60 + a * (30 + (20 - 9 * a) * a)) * y) + 
(10 * a * (60 + a * (30 + (20 - 9 * a) * a)) + 9 * ( - 60 + a * ( - 30 + a * ( - 20 + a * ( - 15 + 8 * a)))) * y) * z) + 
60 * ( - 12 * a**2 - 9 * y * z + 10 * a * (y + z) + a**4 * (6 - 4 * y - 4 * z + 3 * y * z)) * numpy.log(1 - a))

    coefficient = chi2* power2 * (1 + redshift2) * formula
    
    return coefficient

def element2(chi1, chi2, power1, power2, redshift1, redshift2):

    a = 1 - chi1 / chi2
    y = 1 - power1 / power2
    z = 1 - (1 + redshift1) / (1 + redshift2)

    if a == 1:

        formula = (21 - 11 * z) / 12600

    else:

        formula = (1 / (720 * a**4)) * (a * (540 * y * z + 5 * a**3 * ( - 144 + 32 * y + 32 * z - 13 * y * z) + 6 * a**4 * (10 - 5 * y - 5 * z + 3 * y * z) - 60 * a**2 * ( - 12 - 7 * z + y * ( - 7 + 2 * z)) - 
30 * a * (20 * z + y * (20 + 11 * z))) + 60 * ( - 18 * a**3 + a**4 * (6 + y * ( - 2 + z) - 2 * z) + 9 * y * z + 12 * a**2 * (1 + y + z) - 10 * a * (y + z + y * z)) * numpy.log(1 - a))

    coefficient = chi2* power2 * (1 + redshift2) * formula
    
    return coefficient

def element3(chi1, chi2, chi3, chi4, chi5, power1, power2, redshift1, redshift2):

    a = 1 - chi1 / chi2
    b = (chi5-chi3)/(2*chi2)
    c = chi3*numpy.log(chi4/chi3)/(chi4-chi3)-chi5*numpy.log(chi5/chi4)/(chi5-chi4)
    y = 1 - power1 / power2
    z = 1 - (1 + redshift1) / (1 + redshift2)

    if a == 1:

        formula =(1 / 60) * (5 * b + 3 * c - (2 * b + c) * z)

    else:

        formula = (a * ( - 12 * b * y * z + 2 * a**2 * b * ( - 6 + 3 * y + 3 * z - 2 * y * z) + 6 * a * b * (2 * y + 2 * z - y * z) + a**3 * c * (6 - 4 * y - 4 * z + 3 * y * z)) - 12 * b * (a - y) * (a - z) * numpy.log(1 - a)) / (12 * a**3)
    

    coefficient = chi2* power2 * (1 + redshift2) * formula

    return coefficient

def element4(chi1, chi2, chi3, chi4, chi5, power1, power2, redshift1, redshift2):

    a = 1 - chi1 / chi2
    b = (chi5-chi3)/(2*chi2)
    c = chi3*numpy.log(chi4/chi3)/(chi4-chi3)-chi5*numpy.log(chi5/chi4)/(chi5-chi4)
    y = 1 - power1 / power2
    z = 1 - (1 + redshift1) / (1 + redshift2)

    if a == 1:

        formula = (1 / 60) * ( - 2 * c * ( - 6 + z) - 3 * b * ( - 5 + z))

    else:

        formula = (a * (a**3 * c * (6 + y * ( - 2 + z) - 2 * z) + 12 * b * y * z + 2 * a**2 * b * (6 + 3 * y + 3 * z - y * z) - 6 * a * b * (2 * y + (2 + y) * z)) - 
12 * ( - 1 + a) * b * (a - y) * (a - z) * numpy.log(1 - a)) / (12 * a**3)
         

    coefficient = chi2* power2 * (1 + redshift2) * formula

    return coefficient

def element5(chi1, chi2, chi3, power1, power2, redshift1, redshift2):

    a = 1 - chi1 / chi2
    b = chi3 / chi2 - 1
    y = 1 - power1 / power2
    z = 1 - (1 + redshift1) / (1 + redshift2)

    if a == 1:

        formula =  (7 * b * (7 + 5 * b) - 2 * b * (9 + 7 * b) * z + 14 * (1 + b) * ( - 3 + z) * numpy.log(1 + b)) / (840 * b)

    else:

        formula =  - ((1 / (720 * a**4 * b)) * (a * b * (10 * a * (6 * a * ( - 12 + 5 * a**2 + 6 * a * (2 + b)) + 60 * y - a * (66 + 28 * a + 17 * a**2 + 18 * (2 + a) * b) * y) + 
( - 10 * a * ( - 60 + a * (66 + 28 * a + 17 * a**2 + 18 * (2 + a) * b)) + 3 * ( - 180 + a * (30 * (7 + 4 * b) + a * (90 + 60 * b + a * (55 + 39 * a + 40 * b)))) * y) * z) + 
60 * (b * (6 * a**3 * (3 + b) + 10 * a * y - 9 * y * z + a * (10 + 15 * y + 6 * b * y) * z - 2 * a**2 * (6 + 8 * y + 3 * b * y + 8 * z + 3 * b * z) + a**5 * ( - 6 + 4 * y + 4 * z - 3 * y * z) + 
a**4 * (6 - 4 * y - 4 * z + 3 * y * z)) * numpy.log(1 - a) + a**5 * (1 + b) * (6 - 4 * z + y * ( - 4 + 3 * z)) * numpy.log(1 + b))))
        

    coefficient = chi2* power2 * (1 + redshift2) * formula
    
    return coefficient

def element6(chi1, chi2, chi3, power1, power2, redshift1, redshift2):

    a = 1 - chi1 / chi2
    b = chi3 / chi2 - 1
    y = 1 - power1 / power2
    z = 1 - (1 + redshift1) / (1 + redshift2)

    if a == 1:

        formula = (1 / 840) * (175 - 21 * b * ( - 5 + z) - 31 * z + (28 * (1 + b) * ( - 6 + z) * numpy.log(1 + b)) / b)
                                                      
    else:

        formula = (1 / (720 * a**4)) * (60 * ( - 1 + a) * ( - 6 * a**3 * (4 + b) + a**4 * (6 + y * ( - 2 + z) - 2 * z) + 9 * y * z + 6 * a**2 * (2 + (3 + b) * y + (3 + b) * z) - 
2 * a * (5 * z + y * (5 + 8 * z + 3 * b * z))) * numpy.log(1 - a) + a * ( - 540 * y * z + a**4 * ( - 600 + y * (130 - 53 * z) + 130 * z) + 30 * a * (20 * z + y * (20 + 41 * z + 12 * b * z)) + 
5 * a**3 * (360 + 88 * y + 88 * z - 23 * y * z + 12 * b * (6 + 3 * y + 3 * z - y * z)) - 30 * a**2 * (24 + 46 * z + 12 * b * z + y * (46 + 13 * z + 6 * b * (2 + z))) - 
(60 * a**4 * (1 + b) * (6 + y * ( - 2 + z) - 2 * z) * numpy.log(1 + b)) / b))

    coefficient = chi2* power2 * (1 + redshift2) * formula

    return coefficient

def element7(chi1, chi2, chi3, chi4, power1, power2, redshift1, redshift2):

    a = 1 - chi1 / chi2
    b = (chi4-chi3)/(2*chi2)
    c = chi3*numpy.log(chi4/chi3)/(chi4-chi3)-1
    y = 1 - power1 / power2
    z = 1 - (1 + redshift1) / (1 + redshift2)

    if a == 1:

        formula = (1 / 60) * (5 * b + 3 * c - (2 * b + c) * z)

    else: 

        formula =   (a * ( - 12 * b * y * z + 2 * a**2 * b * ( - 6 + 3 * y + 3 * z - 2 * y * z) + 6 * a * b * (2 * y + 2 * z - y * z) + a**3 * c * (6 - 4 * y - 4 * z + 3 * y * z)) - 12 * b * (a - y) * (a - z) * numpy.log(1 - a)) / (12 * a**3)

    coefficient = chi2* power2 * (1 + redshift2) * formula

    return coefficient

def element8(chi1, chi2, chi3, chi4, power1, power2, redshift1, redshift2):

    a = 1 - chi1 / chi2
    b = (chi4-chi3)/(2*chi2)
    c = chi3*numpy.log(chi4/chi3)/(chi4-chi3)-1
    y = 1 - power1 / power2
    z = 1 - (1 + redshift1) / (1 + redshift2)

    if a == 1:

        formula = (1 / 60) * ( - 2 * c * ( - 6 + z) - 3 * b * ( - 5 + z))
        
    else:

        formula = (a * (a**3 * c * (6 + y * ( - 2 + z) - 2 * z) + 12 * b * y * z + 2 * a**2 * b * (6 + 3 * y + 3 * z - y * z) - 6 * a * b * (2 * y + (2 + y) * z)) - 
12 * ( - 1 + a) * b * (a - y) * (a - z) * numpy.log(1 - a)) / (12 * a**3)

    coefficient = chi2* power2 * (1 + redshift2) * formula

    return coefficient



def element(n, i , j, chi_grid, power_grid, redshift_grid):

    ell_size = power_grid.shape[0] - 1
    grid_size = chi_grid.shape[0] - 1

    if (i < n < grid_size) | (j < n < grid_size):
        
        elements = numpy.zeros(ell_size + 1)

    elif (n == j < grid_size) & (n == i < grid_size):

        elements = element1(chi_grid[n], chi_grid[n + 1], power_grid[:,n], power_grid[:,n + 1], redshift_grid[n], redshift_grid[n + 1])

    elif (n == j < grid_size) & (n + 1 == i < grid_size) :

        elements = element2(chi_grid[n], chi_grid[n + 1], power_grid[:,n], power_grid[:,n + 1], redshift_grid[n], redshift_grid[n + 1])

    elif (n + 1 < j < grid_size) & (n == i < grid_size):
        
        elements = element3(chi_grid[n], chi_grid[n + 1], chi_grid[j - 1], chi_grid[j], chi_grid[j + 1], power_grid[:,n], power_grid[:,n + 1], redshift_grid[n], redshift_grid[n + 1])
       
    elif (n + 1 < j < grid_size) & (n + 1 == i < grid_size):

        elements = element4(chi_grid[n], chi_grid[n + 1], chi_grid[j - 1], chi_grid[j], chi_grid[j + 1], power_grid[:,n], power_grid[:,n + 1], redshift_grid[n], redshift_grid[n + 1])

    elif (n + 1 == j < grid_size) & (n == i < grid_size):

        elements = element5(chi_grid[n], chi_grid[n + 1], chi_grid[n + 2], power_grid[:,n], power_grid[:,n + 1], redshift_grid[n], redshift_grid[n + 1])

    elif (n + 1 == j < grid_size) & (n + 1 == i < grid_size):

        elements = element6(chi_grid[n], chi_grid[n + 1], chi_grid[n + 2], power_grid[:,n], power_grid[:,n + 1], redshift_grid[n], redshift_grid[n + 1])
        
    elif (j == grid_size) & (n == i < grid_size):
        
        elements = element7(chi_grid[n], chi_grid[n + 1], chi_grid[grid_size - 1], chi_grid[grid_size], power_grid[:,n], power_grid[:,n + 1], redshift_grid[n], redshift_grid[n + 1])
       
    elif (j == grid_size) & (n + 1 == i < grid_size):

        elements = element8(chi_grid[n], chi_grid[n + 1], chi_grid[grid_size - 1], chi_grid[grid_size], power_grid[:,n], power_grid[:,n + 1], redshift_grid[n], redshift_grid[n + 1])

    else: 

        elements = numpy.zeros(ell_size + 1)

    return elements

def function(amplitude, chi_grid, power_grid, redshift_grid):
    
    grid_size = chi_grid.shape[0] - 1
    ell_size = power_grid.shape[0] - 1
    coefficients = numpy.zeros((grid_size + 1, grid_size + 1, ell_size + 1))
    
    for n in range(grid_size):
        
        for i in range(n, grid_size + 1):
            
            for j in range(n, grid_size + 1):
                
                coefficients[i, j, :] = coefficients[i, j, :] + element(n, i, j, chi_grid, power_grid, redshift_grid)
    coefficients = amplitude * coefficients           
    return coefficients
