Algoritmos desenvolvidos durante o semestre de 1-2025 na disciplina de Redes Neurais.
Cada algoritmo mostra uma abordagem de um tipo de rede neural. Segue a descrição de cada projeto


Projeto: Modelo Neural Não Supervisionado com Mapas Auto-Organizáveis (Projeto 2)
Este projeto explora a aplicação de Modelos Neurais Não Supervisionados, especificamente os Mapas Auto-Organizáveis (Self-Organizing Maps - SOM), para a detecção de padrões e agrupamento de dados em diferentes conjuntos de dados. O objetivo é avaliar a capacidade do SOM em organizar tanto dados sintéticos de baixa dimensão quanto dados complexos de alta dimensão, como imagens de faces. 

Datasets Utilizados
Foram selecionados dois conjuntos de dados para a análise:

Make Moons: Um conjunto de dados sintético que gera duas "luas" entrelaçadas. É ideal para visualizar a capacidade do modelo em separar dados que não são linearmente separáveis. A amostra utilizada contém 250 pontos com um ruído de 0.05.

Olivetti Faces: Um conjunto de dados de alta dimensão contendo 400 imagens de faces (64x64 pixels) de 40 pessoas diferentes. O desafio aqui é verificar se o SOM consegue agrupar as faces por características como identidade, expressão facial ou pose.

Metodologia
O principal modelo utilizado foi o 

MiniSom, uma implementação de Mapas Auto-Organizáveis em Python. O processo consistiu em:
Pré-processamento: Os dados do make_moons foram normalizados utilizando MinMaxScaler para garantir que todas as features tivessem a mesma escala.
Treinamento do SOM:
Para o make_moons, uma rede SOM de 15x15 neurônios foi treinada por 500 épocas;
Para o Olivetti Faces, o mapa foi expandido para 20x20 neurônios para acomodar a maior variabilidade dos dados, e o treinamento foi realizado por 2000 épocas.

Análise de Resultados: U-Matrix (Matriz-U): Foi utilizada para visualizar as distâncias entre os neurônios da rede, ajudando a identificar as fronteiras entre os clusters.

Mapa de Ativação: Mostra quais neurônios foram ativados por quais dados, permitindo a visualização dos agrupamentos formados. No caso das faces, as próprias imagens foram projetadas no mapa para uma análise mais intuitiva.

Erro de Quantização: Foi calculado para cada amostra do dataset de faces para identificar outliers, ou seja, imagens que não se encaixam bem em nenhum cluster formado pela rede.

Resultados
Análise em Dados Sintéticos (make_moons)
O SOM conseguiu mapear a estrutura não-linear do dataset make_moons. A U-Matrix e o Mapa de Ativação (páginas 3 e 4 do documento) demonstram uma clara separação entre os dois grupos, indicando que o modelo foi capaz de aprender a topologia dos dados e agrupar corretamente as duas "luas".

Análise em Dados de Alta Dimensão (Olivetti Faces)
O modelo organizou as 400 imagens de faces em um grid 20x20 de forma coerente. O mapa de ativação (página 7) mostra que faces similares (seja da mesma pessoa ou com poses e expressões parecidas) foram agrupadas em neurônios vizinhos. Além disso, a análise do erro de quantização (páginas 8 e 9) permitiu identificar as imagens que mais se desviam do padrão geral do dataset, funcionando como um método para detecção de anomalias.

Dependências

Para executar este projeto, as seguintes bibliotecas Python são necessárias:
minisom 
numpy 
matplotlib 
scikit-learn 
É possível instalá-las utilizando o pip.



Projeto: Reconhecimento de Atividade Humana com Redes GRU (Projeto 4)

Este projeto implementa uma Rede Neural Recorrente (RNN), especificamente uma Gated Recurrent Unit (GRU), para a tarefa de classificação de séries temporais. O objetivo é classificar seis atividades humanas com base em dados de sensores inerciais de smartphones, utilizando o dataset UCI HAR (Human Activity Recognition Using Smartphones).

Visão Geral do Projeto
O modelo foi treinado para distinguir entre as seguintes seis atividades:
Caminhar (WALKING)
Subir Escadas (WALKING_UPSTAIRS)
Descer Escadas (WALKING_DOWNSTAIRS)
Sentar (SITTING)
Ficar de Pé (STANDING)
Deitar (LAYING)

A implementação abrange desde o download e pré-processamento dos dados brutos até a construção, treinamento e avaliação de um modelo GRU de duas camadas.

Dataset
O conjunto de dados utilizado é o UCI HAR Dataset, que contém registros de 30 voluntários realizando as atividades mencionadas. Os dados foram capturados por meio de acelerômetro e giroscópio de um smartphone, resultando em uma série temporal de 9 features por janela de tempo.

O script carrega os dados diretamente do repositório da UCI, separando-os em conjuntos de treino e teste. Uma análise exploratória mostrou que as classes de atividades estão bem balanceadas no conjunto de dados, o que é ideal para o treinamento.

Metodologia
Carregamento dos Dados: Os dados brutos de sinais inerciais são baixados e montados em arrays NumPy tridimensionais, com formato (amostras, janelas_de_tempo, features).

Pré-processamento: Os rótulos das atividades, originalmente de 1 a 6, são ajustados para o intervalo de 0 a 5 e, em seguida, convertidos para o formato one-hot encoding para serem compatíveis com a função de perda categorical_crossentropy.

Arquitetura do Modelo: Foi construído um modelo sequencial em Keras com a seguinte arquitetura:
Camada de Entrada (Input) com shape (128, 9).
Camada GRU com 100 unidades e return_sequences=True.
Camada de Dropout com taxa de 0.3 para regularização.
Camada GRU com 100 unidades.
Camada de Dropout com taxa de 0.3.
Camada Densa (Dense) com 100 unidades e ativação 'ReLU'.
Camada de Saída (Dense) com 6 unidades (uma para cada classe) e ativação 'softmax'.
Treinamento: O modelo foi compilado com o otimizador adam e treinado por 20 épocas com um batch size de 64.

Resultados e Análise
O modelo alcançou uma performance robusta no conjunto de teste:
Acurácia no Teste: 90.23%
Perda no Teste: 0.3613
A análise da matriz de confusão e do relatório de classificação revelou os seguintes pontos:

Pontos Fortes: O modelo demonstrou excelente capacidade de distinguir entre atividades dinâmicas (caminhar, subir e descer escadas), que apresentaram altíssimas taxas de precisão e recall. A atividade de deitar (LAYING) também foi classificada com grande sucesso.

Limitações: A principal dificuldade do modelo foi diferenciar as atividades estáticas SITTING (Sentar) e STANDING (Ficar de Pé). Houve uma confusão bidirecional significativa entre essas duas classes, indicando que os padrões dos sensores para essas posturas são muito sutis e semelhantes.

Conclusão
O modelo GRU se mostrou eficaz para a tarefa de reconhecimento de atividades humanas, aprendendo com sucesso as macro-características dos movimentos. A alta acurácia geral valida a arquitetura escolhida. A limitação em distinguir entre posturas estáticas semelhantes destaca uma área para melhorias futuras, que poderia envolver o uso de janelas temporais maiores ou a extração de features mais detalhadas.

Dependências
Para executar este notebook, são necessárias as seguintes bibliotecas:
numpy
pandas
matplotlib
seaborn
tensorflow.keras
requests
scikit-learn

Projeto: Análise de Espaço Latente com Variational Autoencoders (Projeto 3)
Este projeto explora a implementação e o treinamento de Variational Autoencoders (VAEs) em dois datasets de imagens distintos. O objetivo principal é analisar como os VAEs aprendem a representar dados complexos em um espaço latente de baixa dimensionalidade e visualizar a organização desses dados usando seus rótulos para colorir as projeções.

O projeto foi desenvolvido por Rafael Pires Moreira Silva (ID: 163978).

Datasets Utilizados:
Foram selecionados dois datasets para avaliar o desempenho e a capacidade de representação dos VAEs:


Rock, Paper, Scissors: Um conjunto de dados com imagens de mãos realizando os três gestos do jogo (pedra, papel e tesoura). É um problema com 3 classes visualmente bem definidas. As imagens foram redimensionadas para 128x128 pixels.

KMNIST (Kuzushiji-MNIST): Uma alternativa ao clássico MNIST, contendo 70.000 imagens em escala de cinza de 10 classes de caracteres do alfabeto japonês Hiragana. As imagens têm o tamanho de 28x28 pixels.

Metodologia:
Para cada dataset, um modelo VAE foi construído e treinado com o objetivo de encontrar a topologia mais eficiente. A implementação do VAE foi feita em TensorFlow/Keras e inclui:

Camada de Amostragem (Sampling Layer): Uma camada customizada para gerar pontos no espaço latente a partir da média (z_mean) e da variância logarítmica (z_log_var) aprendidas pelo encoder.


Arquitetura Encoder-Decoder: 
Encoder: Utiliza camadas convolucionais (Conv2D) para mapear a imagem de entrada para os parâmetros do espaço latente.
Decoder: Utiliza camadas convolucionais transpostas (Conv2DTranspose) para reconstruir a imagem original a partir de um ponto no espaço latente.
Função de Custo (Loss Function): O treinamento é otimizado com base em uma função de custo combinada, que inclui:
Loss de Reconstrução: Mede a diferença entre a imagem original e a reconstruída (usando binary_crossentropy).
Loss de Kullback-Leibler (KL): Atua como um regularizador, forçando a distribuição do espaço latente a se aproximar de uma distribuição normal padrão.

Treinamento:
O VAE para o dataset Rock, Paper, Scissors foi treinado com uma dimensão latente de 16 por 30 épocas.
O VAE para o dataset KMNIST foi treinado com uma dimensão latente de 2 por 10 épocas.
Análise do Espaço Latente: Após o treinamento, o encoder de cada modelo foi usado para projetar as imagens de teste no espaço latente, e os resultados foram visualizados em gráficos de dispersão 2D coloridos pelos rótulos.
Rock, Paper, Scissors
Como o espaço latente original tinha 16 dimensões, a técnica de Análise de Componentes Principais (PCA) foi aplicada para reduzir a dimensionalidade para 2D e permitir a visualização.

O gráfico resultante (página 6) mostra que o VAE conseguiu criar agrupamentos que diferenciam os gestos, embora com uma sobreposição considerável entre as classes. O gesto "paper" (ciano) concentra-se na parte superior, "scissors" (verde) se espalha da esquerda para a direita, e "rock" (roxo) tem maior dispersão, concentrando-se na parte inferior direita, mas interpondo-se com as outras classes.

KMNIST
O VAE foi treinado para mapear as imagens diretamente para um espaço latente de 2 dimensões, não necessitando de PCA.
A projeção 2D (página 7) mostra uma clara separação e agrupamento das 10 classes de caracteres, indicando que o VAE aprendeu uma representação latente bem estruturada e significativa, onde cada cluster corresponde a uma classe de caractere Hiragana.

Dependências:
Para executar este projeto, as seguintes bibliotecas são necessárias:

numpy 
matplotlib 
seaborn 
tensorflow 
tensorflow_datasets 
scikit-learn (para PCA) 

Projeto: Comparação de Arquiteturas CNN e MLP para Classificação de Imagens
Este projeto implementa e avalia diversas topologias de Redes Neurais Convolucionais (CNNs) para a tarefa de classificação de imagens, utilizando o dataset KMNIST. Os resultados das CNNs são comparados entre si para determinar a arquitetura mais eficaz. Além disso, o melhor modelo CNN é comparado com uma rede neural Multi-Layer Perceptron (MLP) para destacar as vantagens das arquiteturas convolucionais em tarefas de visão computacional.

Dataset:
O projeto utiliza o dataset 

KMNIST (Kuzushiji-MNIST), uma alternativa desafiadora ao tradicional MNIST. Ele é composto por imagens de caracteres japoneses do alfabeto Hiragana. Os dados são carregados utilizando o torchvision, normalizados e divididos em DataLoaders para treino e teste.

Metodologia:
O fluxo de trabalho consistiu em treinar e avaliar seis arquiteturas de redes neurais diferentes, sendo cinco CNNs e uma MLP. 

Arquiteturas Avaliadas:
Foram implementadas e testadas cinco topologias de CNN distintas para avaliar o impacto de diferentes parâmetros e configurações: 

CNN Modelo 1: Arquitetura base com duas camadas convolucionais (16 e 32 filtros) e duas camadas totalmente conectadas. 

CNN Modelo 2: Uma versão mais profunda do Modelo 1, com mais filtros (32 e 64) nas camadas convolucionais. 

CNN Modelo 3: Utiliza kernels de convolução maiores (5x5) e uma camada densa com mais neurônios (256). 

CNN Modelo 4: Uma arquitetura ainda mais profunda com três camadas convolucionais. 

CNN Modelo 5: Modelo com alta capacidade, duas camadas convolucionais e uma camada densa maior (512 neurônios), além de uma camada de Dropout (com taxa de 0.5) para regularização. 

Para comparação, um modelo MLP com três camadas totalmente conectadas (512, 256 e 10 neurônios) também foi implementado e treinado. 

Treinamento e Avaliação:

Treinamento: Todos os modelos foram treinados por 10 épocas utilizando o otimizador Adam e a função de perda CrossEntropyLoss. 

Avaliação: O desempenho foi medido pela acurácia no conjunto de teste. 

Análise: Para os dois melhores modelos CNN, a matriz de confusão foi calculada para uma análise de erro mais detalhada. O número de parâmetros treináveis de cada modelo também foi comparado. 

Resultados e Análise
Desempenho dos Modelos CNN
As cinco arquiteturas de CNN apresentaram excelente desempenho, superando a linha de base da MLP. A acurácia final para cada modelo no conjunto de teste foi: 

CNN Modelo 1: 94.54%
CNN Modelo 2: 95.58% 
CNN Modelo 3: 95.72% 
CNN Modelo 4: 95.62% 
CNN Modelo 5: 96.22% (Melhor Desempenho) 

O CNN Modelo 5 se destacou, provavelmente devido à sua camada densa de maior capacidade combinada com a regularização de Dropout, que ajuda a evitar o superajuste. O CNN Modelo 2 foi o segundo melhor, com uma acurácia de 95.17%. A diferença de performance entre os melhores modelos foi mínima , e suas matrizes de confusão comprovaram a robustez, mostrando erros esparsos sem um ponto fraco evidente. 

Comparação: CNN vs. MLP
A análise comparativa entre a melhor CNN e a MLP revelou uma superioridade clara da abordagem convolucional.

Modelo	Acurácia:
Número de Parâmetros Melhor CNN (Modelo 5 - 96.22% 1.630.090 
MLP - 90.25% - 535.818 

Exportar para as Planilhas
A CNN superou a MLP em quase 10% em acurácia, mesmo tendo um número maior de parâmetros. Isso confirma que as CNNs são inerentemente mais adequadas para tarefas de visão computacional, pois são projetadas para processar a estrutura espacial das imagens, ao contrário das MLPs, que tratam as imagens como vetores unidimensionais e perdem essa informação contextual. 

Conclusão
O estudo demonstra a superioridade das arquiteturas convolucionais para a classificação de imagens no dataset KMNIST. A escolha da topologia, como o uso de camadas de Dropout e a capacidade das camadas densas, é crucial para maximizar o desempenho. As CNNs provaram ser a ferramenta mais adequada devido à sua capacidade inata de extrair características hierárquicas e espaciais dos dados. 

Dependências:
Para executar este projeto, são necessárias as seguintes bibliotecas:

torch e torchvision
numpy
itertools
matplotlib
scikit-learn
