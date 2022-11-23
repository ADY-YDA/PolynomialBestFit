import numpy as np
import csv
import argparse
import os
from pandas import DataFrame
import math

# parser
parser = argparse.ArgumentParser(description='Polyhunt project.')

parser.add_argument('--m', type=int, nargs=1, help='polynormial order (or maximum in autofit mode).')
parser.add_argument('--gamma', type=float, default=0.0, help='regularization constant(use a default of 0).')
parser.add_argument('trainPath', type=str, default='./levelOne/A', help='a filepath to the training data.')
parser.add_argument('modelOutput', type=str, default='XXX', help='a filepath where the best fit parameters will be saved. \
    If this is not supplied (please input XXX in place of normal filepath), then no model parameters are outputted.')
parser.add_argument('--autofit', action='store_true', help='flag will engage order sweeping loop. \
    When flag is not supplied, program will simply fit a polynomial of the given order and parameters.\
    In either case, best fit model specified by modelOutput path, and print the RMSE/order \
    information to the screen.')
parser.add_argument('--shuffle', action='store_true',
                    help='adding flag will shuffle the dataset used before running the k-folds.')
parser.add_argument('--numFolds', type=int, default=10, help='the number of folds to use for cross validation.')
parser.add_argument('--max', action='store_true',
                    help='adding flag will toggle to using max RMSE across kfolds testing (True) instead of default max RMSE (False)')

# --m determines the polynomial order. If autofit is not turned on, then one evaluation with order = m will be run. If autofit is turned on, the given integer is used to test the k-folds. For example, a --m value of 20 goes through every polynomial order from 0 to 20.
# --gamma is the regularization constant that is used in the 3.28 equation. If no gamma value is given, the default value is 0.
# --autofit is a boolean value that activates the autofit method. Add the flag to turn this feature on.
# --info is a boolean value that determines whether the name and contact information of collaborators will be outputted. Add the flag to turn this feature on.
# --shuffle is a boolean value that determines whether the dataset is shuffled before running k-folds. Add the flag to turn this feature on.
# --numFolds is an integer value that determines the number of folds that are run in k-folds. If no numFolds value is given and autofit is turned on, the default value is 10.
# --max determines whether the model uses max RMSE across testing (True) or the default max RMSE (False). Add the flag to use max RMSE.
# trainPath is a filepath to the training data
# modelOutput is a filepath where the best fit parameters are outputted. If the string ‘XXX’ is put in place of an actual file path, then no output will occur.

args = parser.parse_args()
polynomial_order = args.m[0]
matrix_m = args.m[0] + 1

# load dataset
if os.path.exists(args.trainPath):
    x = np.loadtxt(args.trainPath, delimiter=',')
else:
    print('Given filepath to training data is invalid, please check and try again!')
    quit()

# randomize the values in order to prepare for training
if args.shuffle:
    np.random.shuffle(x)

# gets the first column of the file - the x-axis
input_variables = [x[:, 0]]
# gets the second column of the file - the y-axis
target_variables = [x[:, 1]]

# turns N column vector into N x M matrix with given M, raising each column to respective power
def get_order_matrix(poly_order, input_var):
    order_matrix = np.tile(input_var.transpose(), (1, poly_order + 1))
    powers = np.linspace(0, poly_order, poly_order + 1)
    order_matrix = np.power(order_matrix, powers)
    return order_matrix

# weights calculation given design_matrix and target_var
def w(design_matrix, target_vars):
    gamma_matrix = np.identity(len(design_matrix[0])) * args.gamma
    weights = np.linalg.pinv(gamma_matrix + design_matrix.transpose().dot(design_matrix)).dot(
        design_matrix.transpose()).dot(np.array(target_vars).transpose())
    return weights

# RMSE error calculation function given predicted and true values
def rmse(predicted, true):
    error = math.sqrt(np.mean(np.power(np.subtract(predicted, true), 2)))
    return error

# run k-fold segment with given m order
def k_fold_cross_val(poly):
    # 6, 7, 3 for X, Y, and Z respectively
    # splits the array into k
    new_array = np.array_split(np.array(x), args.numFolds)
    error_store = np.array([])
    for index in range(len(new_array)):
        array_copy = new_array.copy()
        testing_cases = array_copy[index]
        training_cases = np.concatenate(np.delete(array_copy, index, 0))
        # generate weights using the training cases only
        weight_values = w(get_order_matrix(poly, np.array([training_cases[:][:, 0]])),
                          np.array([training_cases[:][:, 1]]))
        # predict values based on given m order, with test cases x
        predicted_values = np.sum(np.multiply(get_order_matrix(poly, np.array([testing_cases[:][:, 0]])), np.tile(np.array(weight_values), (1, len(testing_cases))).transpose()), axis=1)
        error_store = np.append(error_store, rmse(predicted_values, np.array(testing_cases[:][:, 1])))
    # print(error_store)
    # average the errors for each fold and return that or return the max error
    if args.max:
        return np.max(error_store)
    else:
        return np.mean(error_store)

# Cross-Validation
def compare_error():
    # create a dict where the m_values is the key and the rmse is the respective value
    compare = {}
    # loop for M values stated in the argsparse
    for m_value in range(polynomial_order):
        compare[m_value] = k_fold_cross_val(m_value)
    # we sort the values of the dict in order to from smallest to largest
    sorted_compare = sorted(compare.items(), key=lambda sort: sort[1])
    # we then create a list from the keys of the dict and take the smallest key or the first value in the list
    # print(compare)
    return sorted_compare[0]

error = 0.0;
if not args.autofit:
    # print(get_order_matrix(polynomial_order, np.array(input_variables)))
    w = w(get_order_matrix(polynomial_order, np.array(input_variables)), target_variables)
    predicted_values = np.sum(np.multiply(get_order_matrix(polynomial_order, np.array(input_variables)), np.tile(np.array(w), (1, len(input_variables))).transpose()), axis=1)
    error = rmse(predicted_values, np.array(target_variables))
else:
    optimal_order = int(compare_error()[0])
    # print(optimal_order)
    w = w(get_order_matrix(optimal_order, np.array(input_variables)), target_variables)
    predicted_values = np.sum(np.multiply(get_order_matrix(optimal_order, np.array(input_variables)), np.tile(np.array(w), (1, len(input_variables))).transpose()), axis=1)
    error = rmse(predicted_values, np.array(target_variables))

# program print out
print(args.trainPath)
if args.autofit:
    polynomial_order = optimal_order
    print("optimal m = %d" % (optimal_order))
    if args.shuffle:
        print("shuffle = True")
    else:
        print("shuffle = False")
    if args.max:
        print("using max RMSE")
    else:
        print("using mean RMSE")
    print("k-fold value = %d" % (args.numFolds))
else:
    print("given m = %d" % (polynomial_order))
print("given gamma = %f" % (args.gamma))
print("RMSE error = %f " % (error))
print("predicted weights: ")
print(DataFrame(w).to_string(header=False))

# export to modelOutput if path is given
if args.modelOutput != 'XXX':
    np.savetxt(args.modelOutput, w, header="m = %d\ngamma = %f" % (polynomial_order, args.gamma))
