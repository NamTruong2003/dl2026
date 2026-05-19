import math
import os
import json
def sigmoid(z):
    return 1 / (1 + math.exp(-z))
def readInput(file):
    with open(file,"r") as file:
        lines = [line.strip() for line in file if line.strip()]
    
    n_layer = lines[0]
    n_neuron = lines[1:]
    return n_layer,n_neuron
def matrixMultiply(matrix_a, matrix_b):
    A = matrix_a
    B = matrix_b

    is_A_1d = not isinstance(A, list) or (len(A) > 0 and not isinstance(A[0], list))
    is_B_1d = not isinstance(B, list) or (len(B) > 0 and not isinstance(B[0], list))

    if not isinstance(A, list):
        A = [A]
    if not isinstance(B, list):
        B = [B]

    rows_A = len(A)
    cols_A = 1 if is_A_1d else len(A[0])
    
    rows_B = len(B)
    cols_B = 1 if is_B_1d else len(B[0])

    if cols_A != rows_B:
        orig_A = f"{rows_A}" if is_A_1d else f"{rows_A}x{cols_A}"
        orig_B = f"{rows_B}" if is_B_1d else f"{rows_B}x{cols_B}"
        return f"Error: Incompatible dimensions! Cannot multiply ({orig_A}) by ({orig_B})"

    if is_A_1d and is_B_1d:
        result = 0
        for k in range(rows_A):
            result += A[k] * B[k]
        return result

    if is_A_1d:
        result = [0 for _ in range(cols_B)]
        for j in range(cols_B):
            for k in range(rows_A):
                result[j] += A[k] * B[k][j]
        return result

    if is_B_1d:
        result = [0 for _ in range(rows_A)]
        for i in range(rows_A):
            for k in range(cols_A):
                result[i] += A[i][k] * B[k]
        return result

    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    print(f"mul result:{result}")
    return result

def transpose(matrix):
    return [[matrix[row][col] for row in range(len(matrix))] for col in range(len(matrix[0]))]
def matrixAddition(matrix_a, matrix_b):
    A = matrix_a
    B = matrix_b

    is_A_1d = not isinstance(A, list) or (len(A) > 0 and not isinstance(A[0], list))
    is_B_1d = not isinstance(B, list) or (len(B) > 0 and not isinstance(B[0], list))

    if is_A_1d != is_B_1d:
        return "Error: Matrices must have identical dimensions for addition!"

    if is_A_1d:
        if len(A) != len(B):
            return "Error: Matrices must have identical dimensions for addition!"
        result = [0 for _ in range(len(A))]
        for i in range(len(A)):
            result[i] = A[i] + B[i]
        return result

    rows_A = len(A)
    cols_A = len(A[0])
    rows_B = len(B)
    cols_B = len(B[0])

    if rows_A != rows_B or cols_A != cols_B:
        return "Error: Matrices must have identical dimensions for addition!"

    result = [[0 for _ in range(cols_A)] for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_A):
            result[i][j] = A[i][j] + B[i][j]
    return result

def feedForward(input,weights,bias):
    weights_t = transpose(weights)
    mul = matrixMultiply(weights_t,input)
    return matrixAddition(mul,bias)
def active(z_list):
    return [sigmoid(float(x)) for x in z_list]
def parse_nn_config(file_path):
    if not os.path.exists(file_path):
        return None

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip() and not line.strip().endswith('t')]

    if not lines:
        return None

    config = {}
    try:
        num_layers = int(lines[0])
        config['num_layers'] = num_layers

        neurons_per_layer = [int(lines[i]) for i in range(1, num_layers + 1)]
        config['neurons_per_layer'] = neurons_per_layer

        current_line = num_layers + 1
        config['layers_data'] = {}

        for layer_idx in range(1, num_layers):
            num_neurons_prev = neurons_per_layer[layer_idx - 1]
            
            layer_info = {
                'biases': [float(x) for x in lines[current_line].split()],
                'weights': []
            }
            current_line += 1

            for _ in range(num_neurons_prev):
                node_weights = [float(x) for x in lines[current_line].split()]
                layer_info['weights'].append(node_weights)
                current_line += 1

            config['layers_data'][f'Layer_{layer_idx}'] = layer_info

        return config

    except (IndexError, ValueError):
        return None
    