# MultilayerPerceptron with New Structure

Este projeto implementa uma rede neural do tipo **Multilayer Perceptron (MLP)** com uma nova estrutura em C++. A rede inclui uma camada de entrada, uma camada oculta com ativação ReLU e uma camada de saída com ativação Softmax. O algoritmo de retropropagação (backpropagation) é utilizado para treinar a rede em um dataset simples.

## Funcionalidades

- **Ativação ReLU**: A camada oculta utiliza a função de ativação ReLU (Rectified Linear Unit), e sua derivada é usada durante a retropropagação.
- **Ativação Softmax**: A camada de saída usa a função Softmax para fornecer probabilidades das classes de saída.
- **Feedforward e Retropropagação**: Implementa o cálculo feedforward e retropropagação para o treinamento.
- **Inicialização Aleatória de Pesos**: Os pesos são inicializados aleatoriamente utilizando uma distribuição uniforme.
- **Taxa de Aprendizado Ajustável**: A taxa de aprendizado pode ser configurada ao inicializar o objeto MLP.

## Estrutura da Rede Neural

- **Camada de Entrada**: O número de neurônios é definido por `input_size`. Neste exemplo, são usados 3 neurônios de entrada.
- **Camada Oculta**: A camada oculta consiste em 5 neurônios e utiliza a ativação ReLU.
- **Camada de Saída**: Um único neurônio na camada de saída com ativação Softmax.

## Como Funciona

### Feedforward

1. **Ativação da Camada Oculta**: A soma ponderada das entradas é passada pela função de ativação ReLU.
2. **Ativação da Camada de Saída**: A saída da camada oculta é passada para a camada de saída, e a função Softmax é aplicada para produzir probabilidades.

### Retropropagação

1. **Cálculo do Erro**: O erro na saída é calculado comparando a saída prevista com o valor alvo.
2. **Atualização dos Pesos**: Os pesos são atualizados usando o gradiente descendente com base no erro calculado.

## Como Usar

1. Defina `input_size`, `hidden_size`, `output_size`, e `learning_rate` ao criar o objeto MLP.
2. Forneça os dados de treino (`inputs` e `targets`).
3. Treine o modelo por um número especificado de épocas.
4. Teste o modelo com as mesmas entradas ou novas.

## Exemplo de Uso

```cpp
int input_size = 3;  // Tamanho da camada de entrada
int hidden_size = 5; // Tamanho da camada oculta
int output_size = 1; // Tamanho da camada de saída
double learning_rate = 0.01;

MLP mlp(input_size, hidden_size, output_size, learning_rate);

// Dados de treino
std::vector<double> inputs = {0.1, 0.2, 0.7};
std::vector<double> targets = {1.0};

// Treinamento da rede
for (int epoch = 0; epoch < 1000; ++epoch) {
    mlp.train(inputs, targets);
}

// Teste da rede
std::vector<double> output = mlp.forward(inputs);

std::cout << "Saída da rede: " << output[0] << std::endl;
