PARALLELDQN — TREINAMENTO DQN COM AMBIENTES PARALELOS

Este projeto implementa um algoritmo Deep Q-Network (DQN) com paralelismo por múltiplos ambientes, voltado para simulações robóticas do tipo Leader–Follower.
O objetivo principal é aumentar a taxa de coleta de experiências sem modificar a lógica central do aprendizado por reforço.

O código foi estruturado para uso acadêmico e científico, priorizando clareza, modularidade e reprodutibilidade experimental.

PRINCIPAIS FUNCIONALIDADES

Treinamento DQN com múltiplos ambientes paralelos

Uma única rede neural (policy) e uma única target network

Replay Buffer centralizado e compartilhado

Execução paralela apenas nos ambientes, não nas redes

Compatível com simulações robóticas e ROS2/Gazebo

Avaliação pós-treino em múltiplos níveis de dificuldade

Geração automática de CSV e gráficos experimentais

Controle configurável de logs e métricas de tempo

ARQUITETURA GERAL

O paralelismo ocorre exclusivamente na simulação dos ambientes:

Vários ambientes independentes executam episódios simultaneamente

Todas as transições são enviadas para um Replay Buffer central

A DQN é treinada a partir de amostras desse buffer

Fluxo conceitual:

Ambientes paralelos -> Replay Buffer -> DQN (policy) + Target Network

Essa abordagem mantém o algoritmo DQN original, alterando apenas a forma de coleta de experiências.

ESTRUTURA DO CÓDIGO

Arquivo principal:

ParalelDQN.py

Componentes principais:

ReplayBuffer: armazenamento central de transições

DQN: rede neural usada como policy

LeaderFollowerEnv: ambiente robótico customizado

ParallelEnvs: gerenciador de ambientes paralelos

DQNParallelTrainer: lógica de treino paralelo

EvalAgent: wrapper para avaliação greedy

Funções de cenário: easy, medium, hard

Funções de coleta, geração de CSV e gráficos

Função main com o fluxo completo do experimento

REQUISITOS

Python 3.9 ou superior

PyTorch

NumPy

Matplotlib

Pandas

Instalação típica:

pip install torch numpy matplotlib pandas

Recomenda-se o uso de ambiente virtual.

COMO EXECUTAR

Para executar o treinamento paralelo:

python3 ParalelDQN.py

O script realiza automaticamente:

Inicialização dos ambientes paralelos

Treinamento da DQN até o número máximo de episódios

Impressão do progresso a cada N episódios

Impressão do tempo total e tempo médio por episódio

Avaliação pós-treino por nível de dificuldade

Geração de CSV e gráficos

LOGS DURANTE O TREINAMENTO

Durante o treino, o progresso é exibido a cada intervalo configurável de episódios, por exemplo:

[STATUS] Episodes=200/2000 | Eps=0.084 | Buffer=56563

Ao final do treino, são exibidas métricas de tempo:

Total de episódios
Tempo total de treinamento
Tempo médio por episódio

Essas métricas são úteis para análise de desempenho e comparação com versões sequenciais.

AVALIAÇÃO PÓS-TREINO

Após o treinamento, a policy aprendida é avaliada de forma greedy em três cenários:

Easy

Medium

Hard

Para cada cenário, são calculados:

Número de episódios bem-sucedidos

Taxa de sucesso

Motivos de término (ex.: objetivo atingido, colisão, parede)

Exemplo de saída:

Dificuldade: EASY
Sucessos: 468/500
Taxa de sucesso: 0.94

GERAÇÃO DE CSV E GRÁFICOS

Para cada cenário avaliado, o código gera:

Arquivo CSV contendo a trajetória completa do episódio

Gráfico de posição (posição do líder e do seguidor ao longo do tempo)

Gráfico de velocidade (velocidades linear e angular)

Estrutura típica:

logs/easy/episode_easy.csv
logs/easy/episode_easy_pose.png
logs/easy/episode_easy_vel.png

Esses arquivos podem ser usados diretamente em relatórios e artigos.

CONFIGURAÇÃO DO TREINAMENTO

Os principais hiperparâmetros são definidos em uma estrutura de configuração, incluindo:

Número máximo de episódios

Número de ambientes paralelos

Taxa de aprendizado

Fator de desconto

Parâmetros de epsilon-greedy

Frequência de atualização da target network

Frequência de logs

Isso permite ajustes rápidos sem alterar a lógica principal do código.

BOAS PRÁTICAS CIENTÍFICAS

O código segue boas práticas para pesquisa em Aprendizado por Reforço:

Separação clara entre treino e avaliação

Avaliação feita apenas após o treino

Logs controlados e reprodutíveis

Estrutura modular e extensível

Essa organização facilita a escrita de artigos, relatórios PIBIC e extensões futuras.

EXTENSÕES FUTURAS

Possíveis extensões do projeto incluem:

Comparação entre treino sequencial e paralelo

Logging de reward médio em CSV

Geração automática de tabelas LaTeX

Integração com ROS2 e Gazebo

Curriculum learning (easy para hard)

AUTORIA

Projeto desenvolvido para fins acadêmicos e de pesquisa em
Aprendizado por Reforço e Robótica Móvel.
