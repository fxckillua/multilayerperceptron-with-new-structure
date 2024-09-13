#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

// Função de ativação ReLU
double relu(double x) {
	return x > 0 ? x : 0;
}

// Derivada da função ReLU
double relu_derivative(double x) {
	return x > 0 ? 1 : 0;
}

// Função de ativação Softmax
std::vector<double> softmax(const std::vector<double>& z) {
	double max_z = *max_element(z.begin(), z.end());
	std::vector<double> exp_z(z.size());
	double sum_exp_z = 0.0;
    
	for (size_t i = 0; i < z.size(); i++) {
    	exp_z[i] = exp(z[i] - max_z);
    	sum_exp_z += exp_z[i];
	}
    
	for (size_t i = 0; i < z.size(); i++) {
    	exp_z[i] /= sum_exp_z;
	}
    
	return exp_z;
}

class MLP {
private:
	std::vector<std::vector<double>> weights_hidden;
	std::vector<double> biases_hidden;
	std::vector<double> weights_output;
	double bias_output;
	double learning_rate;

	// Inicialização aleatória de pesos e bias
	double random_weight() {
    	static std::mt19937 gen(42);
    	static std::uniform_real_distribution<> dis(-1, 1);
    	return dis(gen);
	}

public:
	MLP(int input_size, int hidden_size, int output_size, double lr)
    	: weights_hidden(hidden_size, std::vector<double>(input_size)), biases_hidden(hidden_size),
      	weights_output(output_size), bias_output(0), learning_rate(lr) {
   	 
    	// Inicializando pesos e bias da camada oculta
    	for (int i = 0; i < hidden_size; i++) {
        	for (int j = 0; j < input_size; j++) {
            	weights_hidden[i][j] = random_weight();
        	}
        	biases_hidden[i] = random_weight();
    	}

    	// Inicializando pesos e bias da camada de saída
    	for (int i = 0; i < output_size; i++) {
        	weights_output[i] = random_weight();
    	}
    	bias_output = random_weight();
	}

	// Feedforward
	std::vector<double> forward(const std::vector<double>& inputs) {
    	std::vector<double> hidden_layer_output(weights_hidden.size());

    	// Processamento da camada oculta
    	for (size_t i = 0; i < weights_hidden.size(); i++) {
        	double activation = biases_hidden[i];
        	for (size_t j = 0; j < inputs.size(); j++) {
            	activation += weights_hidden[i][j] * inputs[j];
        	}
        	hidden_layer_output[i] = relu(activation);
    	}

    	// Processamento da camada de saída
    	double output_activation = bias_output;
    	for (size_t i = 0; i < hidden_layer_output.size(); i++) {
        	output_activation += weights_output[i] * hidden_layer_output[i];
    	}

    	return softmax({output_activation});
	}

	// Backpropagation e atualização dos pesos
	void train(const std::vector<double>& inputs, const std::vector<double>& targets) {
    	// Forward pass
    	std::vector<double> hidden_layer_output(weights_hidden.size());
    	for (size_t i = 0; i < weights_hidden.size(); i++) {
        	double activation = biases_hidden[i];
        	for (size_t j = 0; j < inputs.size(); j++) {
            	activation += weights_hidden[i][j] * inputs[j];
        	}
        	hidden_layer_output[i] = relu(activation);
    	}

    	double output_activation = bias_output;
    	for (size_t i = 0; i < hidden_layer_output.size(); i++) {
        	output_activation += weights_output[i] * hidden_layer_output[i];
    	}

    	std::vector<double> output = softmax({output_activation});

    	// Cálculo do erro da camada de saída
    	double output_error = targets[0] - output[0];

    	// Backpropagation na camada de saída
    	for (size_t i = 0; i < weights_output.size(); i++) {
        	weights_output[i] += learning_rate * output_error * hidden_layer_output[i];
    	}
    	bias_output += learning_rate * output_error;

    	// Backpropagation na camada oculta
    	for (size_t i = 0; i < weights_hidden.size(); i++) {
        	double hidden_error = output_error * weights_output[i] * relu_derivative(hidden_layer_output[i]);
        	for (size_t j = 0; j < weights_hidden[i].size(); j++) {
            	weights_hidden[i][j] += learning_rate * hidden_error * inputs[j];
        	}
        	biases_hidden[i] += learning_rate * hidden_error;
    	}
	}
};

int main() {
	// Configuração do MLP
	int input_size = 3;	// Número de neurônios de entrada
	int hidden_size = 5;   // Número de neurônios na camada oculta
	int output_size = 1;   // Número de neurônios na camada de saída
	double learning_rate = 0.01;

	MLP mlp(input_size, hidden_size, output_size, learning_rate);

	// Dados de treino (Exemplo)
	std::vector<double> inputs = {0.1, 0.2, 0.7};
	std::vector<double> targets = {1.0};  // Exemplo de target

	// Treinamento da rede
	for (int epoch = 0; epoch < 1000; ++epoch) {
    	mlp.train(inputs, targets);
	}

	// Teste da rede
	std::vector<double> output = mlp.forward(inputs);

	std::cout << "Saida da rede: " << output[0] << std::endl;

	return 0;
}