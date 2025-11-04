# Créditos

O código-base do jogo Pac-Man utilizado neste projeto foi adaptado a partir do repositório original de [Devin Leamy](https://github.com/DevinLeamy/Pacman).

# Models e Logs - Pacman DQN

Estes diretórios contém os modelos treinados e os registros de execução (logs) gerados durante o desenvolvimento do agente DQN para o ambiente Pac-Man.

## Versões

- **Versão 1**  
    Corresponde ao agente **antes** da adição das variáveis relacionadas à posição relativa dos fantasmas e ao estado de poder.  
    Nesta versão, o vetor de estado não inclui as distâncias relativas entre o Pac-Man e os fantasmas, nem o valor retornado por `game.isPowerMode()`.\
    Por conta disso, é necessário fazer as modificações necessárias no `pacman_env.py` para poder utilizar os agentes na `main.py`

- **Versão 2**  
    Corresponde ao agente **após** a inclusão da posição relativa dos fantasmas e da variável `game.isPowerMode()` no vetor de estado.
