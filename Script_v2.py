import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import lmfit as lmf
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution

###################### INFO ##############################

#This script assembles an MSCV (microelectrode sampled current voltammogram) from a collection of current transients.
#The second step then performs a non-linear regression to extract kinetic information from the slope of the curve.

#Import data
data = pd.read_csv("SCV raw.csv")

#Remove spaces from column titles
data.columns = data.columns.str.replace(' /', '_')

#define columns
cols = data.columns

#Find column titles
column_headers = list(data.columns.values)

#Select the desired MSCV
t_ms = 100 #ms
t_SCV = str(1+(t_ms/1000))

#Collate date to make the MSCV
column_headers = list(data.columns.values)
SCV = data.query(column_headers[1]+'=='+t_SCV)

#Define expected values
alpha_model = 0.5
E0_model = 0.5
k0_model = 0.01
k0_min = np.divide(k0_model, 2)
k0_max = np.multiply(k0_model, 2)
E0_min = np.subtract(E0_model, 0.5)
E0_max = np.add(E0_model, 0.5)
if alpha_model >= 0.3:
    alpha_min = np.subtract(alpha_model, 0.3)
else:
    alpha_min = 0.0
if alpha_model <= 0.7:
    alpha_max = np.add(alpha_model, 0.3)
else:
    alpha_max = 1.0    

#Define constants for the regression

n = 1.0
F = 96485.0
Do = 0.0000050
Dr = 0.0000050
a = 0.001250
c = 0.0000050
R = 8.314
T = 298.15
pi = np.pi
        
#Calculate sigma values and t-functions for the model (Mahon and Oldham potential step at microelectrode paper)

sigma_o = (Do * t_ms/1000) / np.power(a,2)
sigma_r = (Dr * t_ms/1000) / np.power(a,2)

if sigma_o <=1.281:
    t_function_o = ((1/(np.sqrt(pi*sigma_o))) + 1 + np.sqrt(sigma_o/(4*pi)) - (3*sigma_o)/25 + (3*(np.power(sigma_o,1.5)))/226)
else:
    t_function_o = ((4/pi) + 8/(np.sqrt(np.power(pi,5)*sigma_o)) + (25*np.power(sigma_o,-1.5))/2792 - np.power(sigma_o,-2.5)/3880 - np.power(sigma_o,-3.5)/4500)

if sigma_r <=1.281:
    t_function_r = ((1/(np.sqrt(pi*sigma_r))) + 1 + np.sqrt(sigma_r/(4*pi)) - (3*sigma_r)/25 + (3*(np.power(sigma_r,1.5)))/226)
else:
    t_function_r = ((4/pi) + 8/(np.sqrt(np.power(pi,5)*sigma_r)) + (25*np.power(sigma_r,-1.5))/2792 - np.power(sigma_r,-2.5)/3880 - np.power(sigma_r,-3.5)/4500)

#SCV model - this is the function for the regression - just in notes for my own reference
#kappa = (((ko*a)/(Do*t_function_o))*np.exp((-alpha*n*F*(x-Eo))/(R*T)))
#theta = (1 + ((Do*t_function_o)/(Dr*t_function_r))*np.exp((n*F*(x-Eo))/(R*T))
#i_theo = ((pi*n*F*Do*c*a*t_function_o)/theta)*np.power((1+(pi/(kappa*theta))*((2*kappa*theta + 3*pi)/(4*kappa*theta + 3*pi*pi))),-1)
#full_i_theo = ((pi*n*F*Do*c*a*t_function_o)/(1 + ((Do*t_function_o)/(Dr*t_function_r))*np.exp((n*F*(x-Eo))/R*T)))*np.power((1+(pi/((((ko*a)/(Do*t_function_o))*np.exp((-alpha*n*F*(x-Eo))/(R*T)))*(1 + ((Do*t_function_o)/(Dr*t_function_r))*np.exp((n*F*(x-Eo))/R*T))))*((2*(((ko*a)/(Do*t_function_o))*np.exp((-alpha*n*F*(x-Eo))/(R*T)))*(1 + ((Do*t_function_o)/(Dr*t_function_r))*np.exp((n*F*(x-Eo))/R*T)) + 3*pi)/(4*(((ko*a)/(Do*t_function_o))*np.exp((-alpha*n*F*(x-Eo))/(R*T)))*theta + 3*pi*pi))),-1)

#Call x and y data

xdata = SCV[cols[0]]
ydata = SCV[cols[2]]

#Add line for t_function_o_m - throwback to iterative code

t_function_o_m = t_function_o
t_function_r_m = t_function_r

#Define the function for the regression

def func(x, k_0, E_0, alpha):
    return ((-pi*n*F*Do*c*a*t_function_o_m)/(1 + ((Do*t_function_o_m)/(Dr*t_function_r_m))*(np.exp((n*F*(x-E_0))/(R*T)))))*(np.power((1+(pi/((((k_0*a)/(Do*t_function_o_m))*(np.exp((-alpha*n*F*(x-E_0))/(R*T))))*(1 + ((Do*t_function_o_m)/(Dr*t_function_r_m))*(np.exp((n*F*(x-E_0))/(R*T))))))*(((2*(((k_0*a)/(Do*t_function_o_m))*(np.exp((-alpha*n*F*(x-E_0))/(R*T))))*(1 + ((Do*t_function_o_m)/(Dr*t_function_r_m))*(np.exp((n*F*(x-E_0))/(R*T))))) + (3*pi))/((4*(((k_0*a)/(Do*t_function_o_m))*(np.exp((-alpha*n*F*(x-E_0))/(R*T))))*(1 + ((Do*t_function_o_m)/(Dr*t_function_r_m))*(np.exp((n*F*(x-E_0))/(R*T))))) + (3*pi*pi)))),-1))

#Define the function for the alogirthm to minimize the sum of squared error

def sum_of_squared_error(parameter_tuple):
    val = func(xdata, *parameter_tuple)
    return np.sum((ydata - val)**2.0)

#Build tuple of initial parameter guesses
#initial_parameters = (k0_model, E0_model, alpha_model)

#Function to generate initial guesses and bounds

def generate_initial_parameters(): #initial guesses are called from the above mix/max code lines
    parameter_bounds = []
    parameter_bounds.append([k0_min, k0_max])
    parameter_bounds.append([E0_min, E0_max])
    parameter_bounds.append([alpha_min, alpha_max])

    #seed the numpy random number generator for repeatable results
    result= differential_evolution(sum_of_squared_error, parameter_bounds, seed=3)
    return result.x

#Generate a tuple of initial parameter guesses
initial_parameters = generate_initial_parameters()

#Curve fit test the data
fitted_parameters, pcov = curve_fit(func, xdata, ydata, initial_parameters, bounds=([k0_min, E0_min, alpha_min], [k0_max, E0_max, alpha_max]))
print('k0 = '+str(fitted_parameters[0])+', E0 = '+str(fitted_parameters[1])+', \u03B1 = '+str(fitted_parameters[2]))

#Calculate the error 
model_predictions = func(xdata, *fitted_parameters)
abs_error = model_predictions - ydata

#Plot SCV
fig, (ax) = plt.subplots()
ax.plot(SCV[cols[0]], SCV[cols[2]], 'ro', label='data')
ax.plot(xdata, func(xdata, *fitted_parameters), label='fit')
ax.set_xlabel("E vs SCE / V")
ax.set_ylabel("I / A")

plt.legend()
fig.tight_layout()

plt.show()

print("Done")
