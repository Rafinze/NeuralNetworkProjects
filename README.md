Algoritmos desenvolvidos durante o semestre de 1-2025 na disciplina de Redes Neurais.
Cada algoritmo mostra uma abordagem de um tipo de rede neural. Segue a descrição de cada projeto


Projeto:Modelo Neural Não Supervisionado com Mapas Auto-Organizáveis (SOM)
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


Projeto 3

Projeto 4

Projeto Final
