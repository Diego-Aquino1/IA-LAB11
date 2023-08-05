import numpy as np

# Parámetros iniciales
bias = 0.12
learning_rate = 0.15
weights = np.array([0.12, -0.41, -0.25])

# Datos de entrenamiento para la puerta lógica AND
data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
expected_output = np.array([0, 0, 0, 1])

# Función de activación (Función Escalón Unitario)
def activation_function(x):
    return 1 if x >= 0 else 0

# Proceso de entrenamiento
epochs = 10
output_file = open("perceptron_output.txt", "w")

output_file.write(f"Pesos iniciales: {weights}\n")

for epoch in range(epochs):
    output_file.write(f"Epoca {epoch + 1}\n")
    output_file.write("--------------------\n")

    for i in range(len(data)):
        inputs = np.array([1, data[i][0], data[i][1]])
        af_input = np.dot(weights, inputs)
        estimated_output = activation_function(af_input)
        error = expected_output[i] - estimated_output

        weights += learning_rate * error * inputs
        output_file.write(f"Dato: {data[i]}\n")
        output_file.write(f"Salida esperada: {expected_output[i]}\n")
        output_file.write(f"Salida estimada: {estimated_output}\n")
        output_file.write(f"Error: {error}\n")
        output_file.write(f"Nuevos pesos: {weights}\n")
        output_file.write("\n")
    
    output_file.write("\n")

output_file.close()