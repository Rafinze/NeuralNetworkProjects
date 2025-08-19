Algoritmos desenvolvidos durante o semestre de 1-2025 na disciplina de Redes Neurais.
Cada algoritmo mostra uma abordagem de um tipo de rede neural. Segue a descrição de cada projeto


Projeto: Modelo Neural Não Supervisionado com Mapas Auto-Organizáveis (SOM)
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



Projeto: Reconhecimento de Atividade Humana com Redes GRU
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

Projeto Final
