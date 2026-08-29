<h1 align="center">🤖 2ROBOTSDQN</h1>

<p align="center">
  Aprendizado por reforço para coordenação <em>Leader–Follower</em> entre dois robôs:<br>
  DQN, Dueling DQN e treino com ambientes paralelos.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white">
  <img src="https://img.shields.io/badge/dom%C3%ADnio-RL%20%2B%20Rob%C3%B3tica-8A2BE2">
</p>

## O problema

Dois robôs precisam sair de um ambiente fechado atravessando uma brecha estreita. Um age como **líder**, o outro como **seguidor**. O desafio não é a navegação isolada — é a coordenação: o seguidor precisa manter formação sem colidir com o líder nem com as paredes, e a passagem só comporta um robô por vez.

O ambiente é implementado do zero em `LeaderFollowerEnv`, com sala fechada, porta e posições de partida variáveis, e a política é treinada por Deep Q-Learning sobre o estado contínuo dos dois agentes.

## As três variantes

O repositório compara três implementações do mesmo problema, cada uma em seu diretório:

| Diretório | Script | O que muda | `TrainConfig` | Seed fixa | Cenários easy/medium/hard |
|---|---|---|:---:|:---:|:---:|
| `Normal_DQN/` | `dqn.py` | DQN clássico, ambiente único e sequencial — a linha de base | — | — | ✅ |
| `Duel-DQN/` | `duel_dqn.py` | Arquitetura **Dueling**, separando valor de estado e vantagem: `Q(s,a) = V(s) + (A(s,a) − mean A(s,a))` | ✅ | ✅ | — |
| `Paralel_DQN/` | `parallel_dqn.py` | Coleta de experiência em **N ambientes paralelos** alimentando um replay buffer central | ✅ | ✅ | ✅ |

As variantes não estão no mesmo estágio de maturidade: `parallel_dqn.py` é a mais completa, `dqn.py` é a mais antiga e ainda não recebeu `TrainConfig` nem seed fixa, e `duel_dqn.py` avalia com `evaluate(n_episodes=200)` em vez dos três cenários. Uniformizar isso é pré-requisito para a comparação valer como experimento.

O paralelismo acontece **apenas na simulação dos ambientes** — a rede continua sendo uma só, com um único target network. O algoritmo de aprendizado não muda; muda a taxa de coleta de transições. Isso mantém a comparação honesta: a diferença de desempenho vem do throughput, não de uma mudança de método.

Detalhes da variante paralela estão em [`Paralel_DQN/DETALHES.md`](Paralel_DQN/DETALHES.md).

## Arquitetura da rede (Dueling)

```
entrada → [Linear → LayerNorm → ReLU] × depth ─┬─→ V(s)      ┐
                                               │             ├─→ Q(s,a)
                                               └─→ A(s,a)    ┘
```

Padrão: `hidden=512`, `depth=3`, LayerNorm em cada bloco, dropout opcional.

## Hiperparâmetros

Definidos em `TrainConfig`, em `duel_dqn.py` e `parallel_dqn.py` (em `dqn.py` ainda estão espalhados pelo código):

| Parâmetro | Padrão | |
|---|---|---|
| `gamma` | 0.99 | fator de desconto |
| `lr` | 1e-3 | taxa de aprendizado |
| `batch` | 128 | tamanho do lote |
| `buffer_capacity` / `min_buffer` | 30000 / 10000 | replay buffer e mínimo antes de treinar |
| `eps_start` → `eps_min` | 1.0 → 0.02 | epsilon-greedy, decaimento `0.9993` |
| `num_envs` | 8 | ambientes paralelos |
| `target_sync_every` | 1500 | passos entre sincronizações da target network |
| `max_episodes` | 10000 | episódios de treino |

`num_envs` deve ser ajustado ao hardware — o comentário no código sugere 2–4 para CPU fraca e 12–32 para CPU forte.

## Como executar

```bash
pip install torch numpy matplotlib pandas

python3 Normal_DQN/dqn.py            # linha de base
python3 Duel-DQN/duel_dqn.py         # Dueling DQN
python3 Paralel_DQN/parallel_dqn.py  # ambientes paralelos
```

Cada script executa o fluxo completo: treino até `max_episodes`, avaliação *greedy* pós-treino e geração dos artefatos.

## Saídas

Em `dqn.py` e `parallel_dqn.py`, a política é avaliada em três níveis de dificuldade (`easy`, `medium`, `hard`) — definidos pela distância e pelo ângulo inicial entre líder e seguidor — reportando taxa de sucesso e motivo de término de cada episódio (objetivo atingido, colisão, parede). Em `duel_dqn.py` a avaliação é única, com 200 episódios. Para cada cenário são gerados:

```
logs/easy/episode_easy.csv        trajetória completa do episódio
logs/easy/episode_easy_pose.png   posição do líder e do seguidor no tempo
logs/easy/episode_easy_vel.png    velocidades linear e angular
```

## Reprodutibilidade

`SEED = 42` fixa `torch`, `numpy` e `random` em `duel_dqn.py` e `parallel_dqn.py`. Treino e avaliação são etapas separadas — a avaliação roda com política *greedy*, depois do treino, nunca durante.

## Limitações conhecidas

- O ambiente é uma simulação 2D própria, não física. A integração com ROS2/Gazebo está prevista, mas ainda não implementada.
- Não há checkpoint de pesos: cada execução treina do zero.
- A comparação entre as três variantes ainda não foi consolidada em tabela ou gráfico único.
- Sem testes automatizados.
- As três variantes ainda não compartilham a mesma configuração: `dqn.py` não tem seed fixa nem `TrainConfig`, e `duel_dqn.py` não avalia por nível de dificuldade.

## Próximos passos

- [ ] Integrar com ROS2 e Gazebo
- [ ] Salvar e carregar checkpoints da policy
- [ ] Consolidar a comparação sequencial × paralelo em uma tabela de resultados
- [ ] *Curriculum learning*: progressão automática de `easy` para `hard`
- [ ] Registrar reward médio em CSV ao longo do treino
- [ ] Uniformizar seed, `TrainConfig` e protocolo de avaliação nas três variantes

---

<sub>Projeto acadêmico de pesquisa em Aprendizado por Reforço e Robótica Móvel · <a href="https://www.linkedin.com/in/abel-severo/">LinkedIn</a></sub>
