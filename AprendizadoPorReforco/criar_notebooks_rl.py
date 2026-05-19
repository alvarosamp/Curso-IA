from pathlib import Path
import json
import textwrap


ROOT = Path(__file__).resolve().parent


def clean(text: str) -> str:
    return textwrap.dedent(text).strip() + "\n"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": clean(text),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": clean(text),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.x",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


aula1 = notebook([
    md(
        """
        # Aula 1 - Aprendizado por Reforço na prática

        **Objetivo:** sair da intuição de "um agente aprende por tentativa e erro" e chegar em implementações pequenas de RL em jogos.

        Ao final desta aula, a turma deve conseguir explicar:

        - o que são agente, ambiente, estado, ação, recompensa, episódio e política;
        - por que exploração importa;
        - como uma tabela Q representa "o valor de escolher uma ação em um estado";
        - como aplicar Q-learning em ambientes simples;
        - como o desenho da recompensa muda o comportamento do agente.

        Tudo aqui usa Python puro. Não precisa de `gym`, `pygame`, `numpy` ou `matplotlib`. Se `matplotlib` existir, os gráficos aparecem; se não existir, o notebook imprime resumos em texto.
        """
    ),
    md(
        """
        ## Roteiro sugerido

        1. **10 min:** intuição e vocabulário de RL.
        2. **15 min:** bandit das moedas para falar de exploração vs. aproveitamento.
        3. **25 min:** Q-learning com uma função genérica de treino.
        4. **25 min:** jogo da memória como ambiente discreto.
        5. **30 min:** Flappy Bird simplificado com estado discretizado.
        6. **10 min:** debate: recompensa mal desenhada, limitações e próximos passos.
        """
    ),
    md(
        """
        ## 1. O modelo mental

        Em aprendizado por reforço, o agente não recebe pares prontos do tipo `entrada -> resposta certa`.

        Ele vive um ciclo:

        ```text
        estado atual -> escolhe ação -> ambiente muda -> recebe recompensa -> novo estado
        ```

        A pergunta central é:

        > "Qual ação devo tomar agora para maximizar a soma de recompensas no futuro?"

        Termos importantes:

        - **Estado (`s`)**: o que o agente consegue observar.
        - **Ação (`a`)**: escolha disponível para o agente.
        - **Recompensa (`r`)**: feedback numérico depois da ação.
        - **Política (`π`)**: regra que escolhe ações.
        - **Retorno**: soma das recompensas ao longo do episódio.
        - **Exploração**: testar ações para aprender.
        - **Aproveitamento**: usar o que já parece melhor.
        """
    ),
    code(
        """
        import random
        import math
        from collections import defaultdict, deque

        try:
            import matplotlib.pyplot as plt
            HAS_PLOTS = True
        except Exception:
            HAS_PLOTS = False

        SEED = 7
        random.seed(SEED)

        def moving_average(values, window=100):
            if not values:
                return []
            acc = []
            total = 0.0
            fila = deque()
            for value in values:
                total += value
                fila.append(value)
                if len(fila) > window:
                    total -= fila.popleft()
                acc.append(total / len(fila))
            return acc

        def plot_series(values, title="", ylabel="retorno", window=100):
            media = moving_average(values, window)
            if HAS_PLOTS:
                plt.figure(figsize=(10, 3))
                plt.plot(media)
                plt.title(title)
                plt.xlabel("episódio")
                plt.ylabel(f"{ylabel} médio ({window})")
                plt.grid(alpha=0.25)
                plt.show()
            else:
                inicio = sum(values[:window]) / max(1, min(window, len(values)))
                fim = sum(values[-window:]) / max(1, min(window, len(values)))
                melhor = max(values) if values else 0
                print(title)
                print(f"episódios: {len(values)}")
                print(f"média inicial: {inicio:.3f} | média final: {fim:.3f} | melhor episódio: {melhor:.3f}")

        def argmax_random_tie(values, actions=None):
            if actions is None:
                actions = range(len(values))
            melhor_valor = max(values[a] for a in actions)
            melhores = [a for a in actions if values[a] == melhor_valor]
            return random.choice(melhores)

        def epsilon_greedy_action(Q, state, n_actions, epsilon, valid_actions=None):
            if valid_actions is None:
                valid_actions = list(range(n_actions))
            else:
                valid_actions = list(valid_actions)

            if random.random() < epsilon:
                return random.choice(valid_actions)
            return argmax_random_tie(Q[state], valid_actions)

        def train_q_learning(env, episodes=1000, max_steps=200, alpha=0.2, gamma=0.95,
                             epsilon_start=1.0, epsilon_end=0.05, seed=SEED):
            random.seed(seed)
            Q = defaultdict(lambda: [0.0] * env.n_actions)
            returns = []

            for ep in range(episodes):
                frac = ep / max(1, episodes - 1)
                epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)
                state = env.reset()
                total = 0.0

                for _ in range(max_steps):
                    valid_actions = env.valid_actions() if hasattr(env, "valid_actions") else range(env.n_actions)
                    action = epsilon_greedy_action(Q, state, env.n_actions, epsilon, valid_actions)
                    next_state, reward, done, info = env.step(action)

                    next_valid_actions = env.valid_actions() if hasattr(env, "valid_actions") else range(env.n_actions)
                    best_next = max(Q[next_state][a] for a in next_valid_actions) if not done else 0.0
                    target = reward + gamma * best_next
                    Q[state][action] += alpha * (target - Q[state][action])

                    state = next_state
                    total += reward
                    if done:
                        break

                returns.append(total)

            return Q, returns

        def run_greedy_episode(env, Q, max_steps=80, render=True):
            state = env.reset()
            total = 0.0
            if render and hasattr(env, "render"):
                env.render()

            for step in range(max_steps):
                valid_actions = env.valid_actions() if hasattr(env, "valid_actions") else range(env.n_actions)
                action = argmax_random_tie(Q[state], valid_actions)
                state, reward, done, info = env.step(action)
                total += reward

                if render and hasattr(env, "render"):
                    print(f"passo={step + 1:02d} ação={action} recompensa={reward:.2f}")
                    env.render()

                if done:
                    break

            print(f"retorno total: {total:.2f}")
            return total
        """
    ),
    md(
        """
        ## 2. Aquecimento: bandit das moedas

        Este é o menor "jogo" possível: há algumas máquinas caça-moedas, cada uma com uma chance diferente de pagar recompensa.

        O agente só precisa aprender qual braço puxa mais recompensa em média.

        Não há futuro distante aqui; o episódio acaba depois de uma ação. Mesmo assim, já aparece a tensão central:

        - explorar máquinas pouco testadas;
        - aproveitar a máquina que parece melhor.
        """
    ),
    code(
        """
        class BanditDasMoedas:
            def __init__(self, probabilidades=(0.10, 0.25, 0.55, 0.80)):
                self.probabilidades = list(probabilidades)
                self.n_actions = len(self.probabilidades)

            def reset(self):
                return "inicio"

            def valid_actions(self):
                return range(self.n_actions)

            def step(self, action):
                ganhou = random.random() < self.probabilidades[action]
                reward = 1.0 if ganhou else 0.0
                done = True
                info = {"probabilidade_real": self.probabilidades[action]}
                return "inicio", reward, done, info


        bandit = BanditDasMoedas()
        Q_bandit, retornos_bandit = train_q_learning(
            bandit,
            episodes=800,
            max_steps=1,
            alpha=0.1,
            gamma=0.0,
            epsilon_start=1.0,
            epsilon_end=0.02,
        )

        print("Valores Q aprendidos:")
        for acao, valor in enumerate(Q_bandit["inicio"]):
            print(f"ação {acao}: Q={valor:.3f} | probabilidade real={bandit.probabilidades[acao]:.2f}")

        plot_series(retornos_bandit, "Bandit das moedas", ylabel="recompensa", window=50)
        """
    ),
    md(
        """
        ## 3. A regra do Q-learning

        A tabela Q guarda uma estimativa:

        ```text
        Q[estado][ação] = quão boa parece essa ação nesse estado
        ```

        Depois de agir, o agente atualiza essa estimativa:

        ```text
        novo_Q = Q + α * (alvo - Q)

        alvo = recompensa + γ * melhor_Q_no_próximo_estado
        ```

        Onde:

        - `α` controla o tamanho do passo de aprendizado;
        - `γ` controla o peso do futuro;
        - `epsilon` controla a chance de explorar.
        """
    ),
    md(
        """
        ## 4. Jogo da memória como ambiente de RL

        Agora o agente joga uma versão pequena do jogo da memória.

        Regras:

        - há pares de cartas escondidas;
        - o agente escolhe uma carta por ação;
        - quando escolhe a segunda carta, ganha recompensa se formar par;
        - cartas já vistas entram no estado como "memória" do agente;
        - o episódio termina quando todos os pares foram encontrados.

        Observação didática: este ambiente é pequeno de propósito. Se aumentarmos muito o número de cartas, Q-learning tabular fica rapidamente inviável.
        """
    ),
    code(
        """
        class MemoryGameEnv:
            def __init__(self, pairs=3, shuffle=False):
                self.pairs = pairs
                self.n_cards = pairs * 2
                self.n_actions = self.n_cards
                self.shuffle = shuffle

            def reset(self):
                if self.shuffle:
                    self.deck = [value for value in range(self.pairs) for _ in range(2)]
                    random.shuffle(self.deck)
                else:
                    self.deck = list(range(self.pairs)) + list(range(self.pairs))

                self.known = [-1] * self.n_cards
                self.matched = [False] * self.n_cards
                self.first_pick = -1
                return self._state()

            def _state(self):
                return (tuple(self.known), tuple(self.matched), self.first_pick)

            def valid_actions(self):
                if self.first_pick == -1:
                    return [i for i in range(self.n_cards) if not self.matched[i]]
                return [i for i in range(self.n_cards) if not self.matched[i] and i != self.first_pick]

            def step(self, action):
                if action < 0 or action >= self.n_cards:
                    return self._state(), -1.0, False, {"erro": "ação fora do tabuleiro"}
                if self.matched[action] or action == self.first_pick:
                    return self._state(), -0.7, False, {"erro": "carta inválida"}

                value = self.deck[action]
                self.known[action] = value

                if self.first_pick == -1:
                    self.first_pick = action
                    return self._state(), -0.02, False, {"evento": "primeira carta"}

                previous = self.first_pick
                self.first_pick = -1

                if self.deck[previous] == value:
                    self.matched[previous] = True
                    self.matched[action] = True
                    reward = 1.0
                    done = all(self.matched)
                    if done:
                        reward += 3.0
                    return self._state(), reward, done, {"evento": "par"}

                return self._state(), -0.20, False, {"evento": "erro"}

            def render(self):
                cells = []
                for i in range(self.n_cards):
                    if self.matched[i]:
                        face = f"[{self.deck[i]}]"
                    elif i == self.first_pick:
                        face = f" {self.deck[i]} "
                    elif self.known[i] != -1:
                        face = f"({self.known[i]})"
                    else:
                        face = " ? "
                    cells.append(f"{i}:{face}")
                print("  ".join(cells))


        memoria = MemoryGameEnv(pairs=3, shuffle=False)
        Q_memoria, retornos_memoria = train_q_learning(
            memoria,
            episodes=3500,
            max_steps=30,
            alpha=0.25,
            gamma=0.92,
            epsilon_start=1.0,
            epsilon_end=0.03,
        )

        plot_series(retornos_memoria, "Jogo da memória", ylabel="retorno", window=150)
        print("Episódio guloso depois do treino:")
        run_greedy_episode(MemoryGameEnv(pairs=3, shuffle=False), Q_memoria, max_steps=20)
        """
    ),
    md(
        """
        ### Discussão rápida

        Perguntas boas para fazer em sala:

        - O que aconteceria se a recompensa por errar fosse positiva?
        - O agente realmente "entende" memória ou só aprendeu uma tabela?
        - Por que embaralhar as cartas em todo episódio deixa o problema mais difícil?
        - Qual informação precisa estar no estado para o agente tomar uma boa decisão?
        """
    ),
    md(
        """
        ## 5. Flappy Bird simplificado

        Agora vamos criar um Flappy Bird mínimo:

        - a ave fica em uma coluna fixa;
        - a ação `0` deixa cair;
        - a ação `1` bate asa;
        - há um cano com uma abertura;
        - sobreviver dá uma recompensa pequena;
        - passar pelo cano dá recompensa maior;
        - bater no chão, teto ou cano encerra o episódio.

        O estado é discretizado como:

        ```text
        (altura_da_ave, velocidade, distância_horizontal_do_cano, diferença_para_o_centro_do_vão)
        ```
        """
    ),
    code(
        """
        class MiniFlappyEnv:
            def __init__(self, width=12, height=10, gap_size=3):
                self.width = width
                self.height = height
                self.gap_size = gap_size
                self.bird_x = 2
                self.n_actions = 2

            def reset(self):
                self.bird_y = self.height // 2
                self.velocity = 0
                self.pipe_x = self.width - 1
                self.gap_top = random.randint(1, self.height - self.gap_size - 1)
                self.score = 0
                return self._state()

            def _gap_center(self):
                return self.gap_top + self.gap_size // 2

            def _state(self):
                dx = max(0, self.pipe_x - self.bird_x)
                dy = self._gap_center() - self.bird_y
                dy = max(-4, min(4, dy))
                velocity = max(-2, min(2, self.velocity))
                return (self.bird_y, velocity, dx, dy)

            def valid_actions(self):
                return [0, 1]

            def _collided_with_pipe(self):
                in_gap = self.gap_top <= self.bird_y < self.gap_top + self.gap_size
                return self.pipe_x == self.bird_x and not in_gap

            def step(self, action):
                old_pipe_x = self.pipe_x

                if action == 1:
                    self.velocity = -2
                else:
                    self.velocity += 1
                self.velocity = max(-2, min(2, self.velocity))

                self.bird_y += self.velocity
                self.pipe_x -= 1

                reward = 0.08
                done = False

                if self.bird_y < 0 or self.bird_y >= self.height:
                    return self._state(), -5.0, True, {"evento": "bateu no limite"}

                if self._collided_with_pipe():
                    return self._state(), -5.0, True, {"evento": "bateu no cano"}

                if old_pipe_x >= self.bird_x and self.pipe_x < self.bird_x:
                    self.score += 1
                    reward += 1.0

                if self.pipe_x < 0:
                    self.pipe_x = self.width - 1
                    self.gap_top = random.randint(1, self.height - self.gap_size - 1)

                return self._state(), reward, done, {"score": self.score}

            def render(self):
                rows = []
                for r in range(self.height):
                    chars = []
                    for c in range(self.width):
                        ch = "."
                        if c == self.pipe_x and not (self.gap_top <= r < self.gap_top + self.gap_size):
                            ch = "#"
                        if c == self.bird_x and r == self.bird_y:
                            ch = "B"
                        chars.append(ch)
                    rows.append("".join(chars))
                print("\\n".join(rows))
                print(f"vel={self.velocity} score={self.score}\\n")


        flappy = MiniFlappyEnv()
        Q_flappy, retornos_flappy = train_q_learning(
            flappy,
            episodes=6000,
            max_steps=250,
            alpha=0.18,
            gamma=0.98,
            epsilon_start=1.0,
            epsilon_end=0.04,
        )

        plot_series(retornos_flappy, "Mini Flappy Bird", ylabel="retorno", window=200)
        """
    ),
    code(
        """
        print("Um episódio guloso do Flappy treinado:")
        run_greedy_episode(MiniFlappyEnv(), Q_flappy, max_steps=35, render=True)
        """
    ),
    md(
        """
        ## 6. Exemplo: Q-learning com rede neural

        Em jogos reais, uma tabela `Q[estado][ação]` fica grande demais.

        A ideia por trás de métodos como **DQN** é trocar a tabela por uma função aproximadora:

        ```text
        rede_neural(estado) -> [Q(estado, ação_0), Q(estado, ação_1), ...]
        ```

        Para manter a aula leve, vamos fazer uma ponte didática:

        1. primeiro usamos o Q-learning tabular que já treinou o Flappy;
        2. depois treinamos uma rede neural pequena para imitar essa tabela;
        3. por fim usamos a rede para jogar.

        Isso não é um DQN completo, porque ainda não treinamos a rede diretamente pelas experiências. Mas é o melhor primeiro degrau: a turma vê exatamente o que a rede está substituindo.
        """
    ),
    code(
        """
        class TinyQNetwork:
            def __init__(self, n_inputs, n_hidden, n_outputs, lr=0.015, seed=SEED):
                random.seed(seed)
                self.lr = lr
                self.w1 = [[random.uniform(-0.4, 0.4) for _ in range(n_inputs)] for _ in range(n_hidden)]
                self.b1 = [0.0 for _ in range(n_hidden)]
                self.w2 = [[random.uniform(-0.4, 0.4) for _ in range(n_hidden)] for _ in range(n_outputs)]
                self.b2 = [0.0 for _ in range(n_outputs)]

            def _tanh(self, x):
                return math.tanh(x)

            def forward(self, x):
                hidden = []
                for h, weights in enumerate(self.w1):
                    z = self.b1[h] + sum(w * xi for w, xi in zip(weights, x))
                    hidden.append(self._tanh(z))

                out = []
                for o, weights in enumerate(self.w2):
                    y = self.b2[o] + sum(w * hi for w, hi in zip(weights, hidden))
                    out.append(y)
                return hidden, out

            def predict(self, x):
                return self.forward(x)[1]

            def train_one(self, x, target):
                hidden, out = self.forward(x)
                grad_out = [2 * (out[o] - target[o]) / len(out) for o in range(len(out))]
                old_w2 = [row[:] for row in self.w2]

                for o in range(len(out)):
                    for h in range(len(hidden)):
                        self.w2[o][h] -= self.lr * grad_out[o] * hidden[h]
                    self.b2[o] -= self.lr * grad_out[o]

                for h in range(len(hidden)):
                    grad_h = sum(grad_out[o] * old_w2[o][h] for o in range(len(out)))
                    grad_h *= (1 - hidden[h] ** 2)
                    for i in range(len(x)):
                        self.w1[h][i] -= self.lr * grad_h * x[i]
                    self.b1[h] -= self.lr * grad_h

                return sum((out[o] - target[o]) ** 2 for o in range(len(out))) / len(out)


        def flappy_features(state, env):
            bird_y, velocity, dx, dy = state
            max_dx = max(1, env.width - env.bird_x)
            return [
                (bird_y / (env.height - 1)) * 2 - 1,
                velocity / 2,
                (dx / max_dx) * 2 - 1,
                dy / 4,
            ]


        base_env = MiniFlappyEnv()
        dataset = []
        for state, q_values in Q_flappy.items():
            if any(abs(value) > 1e-9 for value in q_values):
                dataset.append((flappy_features(state, base_env), q_values[:]))

        neural_q = TinyQNetwork(n_inputs=4, n_hidden=10, n_outputs=2, lr=0.012)

        losses = []
        for epoch in range(80):
            random.shuffle(dataset)
            total_loss = 0.0
            for x, target in dataset:
                total_loss += neural_q.train_one(x, target)
            losses.append(total_loss / max(1, len(dataset)))

        print(f"amostras usadas para treinar a rede: {len(dataset)}")
        print(f"loss inicial: {losses[0]:.4f} | loss final: {losses[-1]:.4f}")

        sample_state, sample_target = random.choice(list(Q_flappy.items()))
        sample_prediction = neural_q.predict(flappy_features(sample_state, base_env))
        print("exemplo de estado:", sample_state)
        print("Q da tabela:       ", [round(v, 2) for v in sample_target])
        print("Q da rede neural:  ", [round(v, 2) for v in sample_prediction])


        def run_neural_flappy_episode(policy_net, max_steps=120):
            env = MiniFlappyEnv()
            state = env.reset()
            total = 0.0
            for _ in range(max_steps):
                values = policy_net.predict(flappy_features(state, env))
                action = argmax_random_tie(values)
                state, reward, done, info = env.step(action)
                total += reward
                if done:
                    break
            print(f"episódio com a rede neural: retorno={total:.2f} score={env.score}")
            return total, env.score


        run_neural_flappy_episode(neural_q)

        plot_series(losses, "Rede neural imitando a tabela Q", ylabel="loss", window=5)
        """
    ),
    md(
        """
        ## 7. Fechamento da aula

        Pontos que valem ser reforçados:

        - RL depende muito de uma boa definição de **estado**.
        - A recompensa é o "idioma" usado para dizer ao agente o que importa.
        - Q-learning tabular funciona bem em espaços pequenos, mas escala mal.
        - Jogos maiores costumam exigir aproximação de função, por exemplo redes neurais.
        - DQN é, em essência, Q-learning com uma rede neural aproximando os valores Q.
        - Um agente pode maximizar exatamente a recompensa que você deu e ainda assim fazer algo indesejado.

        Miniatividades:

        1. Mude a penalidade por bater no Flappy de `-5` para `-1`. O comportamento muda?
        2. Aumente o jogo da memória para 4 pares. Quantos estados parecem aparecer?
        3. Treine o Flappy com `gamma=0.5` e depois com `gamma=0.99`. O que muda?
        4. Faça uma política manual para o Flappy: bater asa se estiver abaixo do vão. Compare com Q-learning.
        5. Mude o número de neurônios escondidos da rede de `10` para `3` e depois para `30`. A imitação melhora?
        """
    ),
])


aula2 = notebook([
    md(
        """
        # Aula 2 - Exercício: O Labirinto do Tesouro

        Nesta aula, a turma vai completar um agente que aprende a atravessar um mapa com paredes, tesouros, lava, vento, chave e porta.

        A ideia é transformar Q-learning em uma atividade mais investigativa:

        - primeiro entender o ambiente;
        - depois implementar exploração e atualização da tabela Q;
        - por fim comparar políticas e ajustar recompensas/hiperparâmetros.

        O notebook tem TODOs para estudantes e um gabarito no final para o professor.
        """
    ),
    md(
        """
        ## Regras do ambiente

        Símbolos do mapa:

        - `S`: início;
        - `G`: objetivo final;
        - `#`: parede;
        - `T`: tesouro;
        - `K`: chave;
        - `D`: porta, só passa com a chave;
        - `L`: lava, manda o agente de volta ao início;
        - `W`: vento, empurra o agente uma casa para a direita se possível;
        - `.`: caminho livre.

        Recompensas:

        - cada passo custa um pouco;
        - bater em parede custa mais um pouco;
        - pegar tesouro recompensa;
        - pegar chave recompensa;
        - cair na lava penaliza;
        - chegar ao objetivo com a chave encerra o episódio com grande recompensa.
        """
    ),
    code(
        """
        import random
        import math
        from collections import defaultdict, deque

        try:
            import matplotlib.pyplot as plt
            HAS_PLOTS = True
        except Exception:
            HAS_PLOTS = False

        SEED = 11
        random.seed(SEED)

        ACTIONS = {
            0: (-1, 0),  # cima
            1: (0, 1),   # direita
            2: (1, 0),   # baixo
            3: (0, -1),  # esquerda
        }
        ACTION_NAMES = {0: "↑", 1: "→", 2: "↓", 3: "←"}

        def moving_average(values, window=100):
            if not values:
                return []
            out, total, fila = [], 0.0, deque()
            for value in values:
                total += value
                fila.append(value)
                if len(fila) > window:
                    total -= fila.popleft()
                out.append(total / len(fila))
            return out

        def plot_series(values, title="", ylabel="retorno", window=100):
            media = moving_average(values, window)
            if HAS_PLOTS:
                plt.figure(figsize=(10, 3))
                plt.plot(media)
                plt.title(title)
                plt.xlabel("episódio")
                plt.ylabel(f"{ylabel} médio ({window})")
                plt.grid(alpha=0.25)
                plt.show()
            else:
                inicio = sum(values[:window]) / max(1, min(window, len(values)))
                fim = sum(values[-window:]) / max(1, min(window, len(values)))
                print(title)
                print(f"episódios: {len(values)} | média inicial={inicio:.2f} | média final={fim:.2f} | melhor={max(values):.2f}")

        def argmax_random_tie(values):
            best = max(values)
            options = [i for i, value in enumerate(values) if value == best]
            return random.choice(options)
        """
    ),
    code(
        """
        class TreasureMazeEnv:
            MAP = [
                "##########",
                "#S..T...K#",
                "#.##.##..#",
                "#....L...#",
                "#T.##.##D#",
                "#....W.#G#",
                "##########",
            ]

            def __init__(self):
                self.grid = [list(row) for row in self.MAP]
                self.height = len(self.grid)
                self.width = len(self.grid[0])
                self.n_actions = 4
                self.start = self._find("S")
                self.goal = self._find("G")
                self.key_pos = self._find("K")
                self.treasures = self._find_all("T")

            def _find(self, symbol):
                for r, row in enumerate(self.grid):
                    for c, value in enumerate(row):
                        if value == symbol:
                            return (r, c)
                raise ValueError(f"símbolo {symbol!r} não encontrado")

            def _find_all(self, symbol):
                positions = []
                for r, row in enumerate(self.grid):
                    for c, value in enumerate(row):
                        if value == symbol:
                            positions.append((r, c))
                return positions

            def reset(self):
                self.pos = self.start
                self.has_key = False
                self.treasure_mask = 0
                self.steps = 0
                return self._state()

            def _state(self):
                return (self.pos[0], self.pos[1], int(self.has_key), self.treasure_mask)

            def valid_actions(self):
                return range(self.n_actions)

            def _tile(self, pos):
                r, c = pos
                return self.grid[r][c]

            def _is_blocked(self, pos):
                r, c = pos
                if r < 0 or r >= self.height or c < 0 or c >= self.width:
                    return True
                if self.grid[r][c] == "#":
                    return True
                if self.grid[r][c] == "D" and not self.has_key:
                    return True
                return False

            def step(self, action):
                self.steps += 1
                dr, dc = ACTIONS[action]
                r, c = self.pos
                candidate = (r + dr, c + dc)

                reward = -0.03
                done = False
                info = {}

                if self._is_blocked(candidate):
                    reward -= 0.12
                    candidate = self.pos
                    info["evento"] = "bloqueado"

                self.pos = candidate
                tile = self._tile(self.pos)

                if tile == "W":
                    pushed = (self.pos[0], self.pos[1] + 1)
                    if not self._is_blocked(pushed):
                        self.pos = pushed
                        reward += 0.03
                        info["evento"] = "vento"

                tile = self._tile(self.pos)

                if tile == "K" and not self.has_key:
                    self.has_key = True
                    reward += 0.80
                    info["evento"] = "chave"

                if tile == "T":
                    idx = self.treasures.index(self.pos)
                    bit = 1 << idx
                    if not (self.treasure_mask & bit):
                        self.treasure_mask |= bit
                        reward += 1.00
                        info["evento"] = "tesouro"

                if tile == "L":
                    reward -= 2.00
                    self.pos = self.start
                    info["evento"] = "lava"

                if tile == "G":
                    if self.has_key:
                        collected = bin(self.treasure_mask).count("1")
                        reward += 4.00 + collected
                        done = True
                        info["evento"] = "objetivo"
                    else:
                        reward -= 0.50
                        info["evento"] = "sem_chave"

                return self._state(), reward, done, info

            def render(self, Q=None):
                state = self._state()
                policy_action = None
                if Q is not None and state in Q:
                    policy_action = argmax_random_tie(Q[state])

                for r, row in enumerate(self.grid):
                    chars = []
                    for c, tile in enumerate(row):
                        pos = (r, c)
                        ch = tile
                        if pos == self.pos:
                            ch = "A" if policy_action is None else ACTION_NAMES[policy_action]
                        elif tile == "T":
                            idx = self.treasures.index(pos)
                            if self.treasure_mask & (1 << idx):
                                ch = "."
                        elif tile == "K" and self.has_key:
                            ch = "."
                        chars.append(ch)
                    print("".join(chars))
                print(f"estado={state} chave={self.has_key} passos={self.steps}\\n")


        env = TreasureMazeEnv()
        state = env.reset()
        env.render()
        """
    ),
    md(
        """
        ## Exercício 1 - Política epsilon-greedy

        Complete a função abaixo.

        Com probabilidade `epsilon`, ela deve escolher uma ação aleatória.

        Caso contrário, deve escolher uma das ações de maior valor em `Q[state]`.
        """
    ),
    code(
        """
        def epsilon_greedy(Q, state, n_actions, epsilon):
            # TODO estudante:
            # 1. Se o estado ainda não existe em Q, o defaultdict já cria uma lista de zeros.
            # 2. Com probabilidade epsilon, escolha uma ação aleatória.
            # 3. Caso contrário, escolha a ação de maior valor, desempate aleatoriamente.
            raise NotImplementedError("implemente epsilon_greedy")
        """
    ),
    md(
        """
        ## Exercício 2 - Atualização Q-learning

        Complete a função de atualização:

        ```text
        alvo = reward + gamma * max_a Q[next_state][a]
        Q[state][action] = Q[state][action] + alpha * (alvo - Q[state][action])
        ```

        Se `done=True`, o futuro deve valer zero.
        """
    ),
    code(
        """
        def q_learning_update(Q, state, action, reward, next_state, done, alpha, gamma):
            # TODO estudante:
            # 1. Calcule o melhor valor futuro.
            # 2. Calcule o alvo.
            # 3. Atualize Q[state][action].
            # 4. Retorne Q[state][action].
            raise NotImplementedError("implemente q_learning_update")
        """
    ),
    md(
        """
        ## Exercício 3 - Treinamento

        Agora complete o laço de treino.

        Dicas:

        - `epsilon` deve diminuir ao longo dos episódios;
        - em cada passo, escolha ação, chame `env.step(action)` e atualize Q;
        - guarde o retorno total de cada episódio em uma lista.
        """
    ),
    code(
        """
        def train_agent(env, episodes=3000, max_steps=160, alpha=0.20, gamma=0.96,
                        epsilon_start=1.0, epsilon_end=0.05, seed=SEED):
            # TODO estudante:
            # 1. Crie Q como defaultdict(lambda: [0.0] * env.n_actions).
            # 2. Para cada episódio, resete o ambiente.
            # 3. Reduza epsilon linearmente de epsilon_start até epsilon_end.
            # 4. Rode até max_steps ou done=True.
            # 5. Retorne Q e a lista de retornos.
            raise NotImplementedError("implemente train_agent")
        """
    ),
    md(
        """
        ## Testes rápidos

        Rode esta célula depois de implementar os três exercícios.
        """
    ),
    code(
        """
        try:
            teste_Q = defaultdict(lambda: [0.0] * 4)
            teste_Q["s"] = [0.0, 1.0, 1.0, -1.0]
            escolhas = {epsilon_greedy(teste_Q, "s", 4, epsilon=0.0) for _ in range(30)}
            assert escolhas <= {1, 2}, escolhas

            valor = q_learning_update(teste_Q, "s", 0, reward=1.0, next_state="s2",
                                      done=True, alpha=0.5, gamma=0.9)
            assert abs(valor - 0.5) < 1e-9, valor

            print("testes passaram")
        except NotImplementedError as exc:
            print(f"ainda falta: {exc}")
        """
    ),
    md(
        """
        ## Treine seu agente

        Depois dos TODOs, esta célula deve treinar uma política para o labirinto.
        """
    ),
    code(
        """
        try:
            env = TreasureMazeEnv()
            Q, returns = train_agent(
                env,
                episodes=4500,
                max_steps=180,
                alpha=0.20,
                gamma=0.97,
                epsilon_start=1.0,
                epsilon_end=0.04,
            )
            plot_series(returns, "Labirinto do Tesouro", ylabel="retorno", window=200)
            print(f"estados visitados: {len(Q)}")
        except NotImplementedError as exc:
            print(f"implemente antes de treinar: {exc}")
        """
    ),
    code(
        """
        def run_episode(env, Q, max_steps=80, render=True):
            state = env.reset()
            total = 0.0
            if render:
                env.render(Q)
            for step in range(max_steps):
                action = argmax_random_tie(Q[state])
                state, reward, done, info = env.step(action)
                total += reward
                if render:
                    print(f"passo={step + 1:02d} ação={ACTION_NAMES[action]} reward={reward:.2f} info={info}")
                    env.render(Q)
                if done:
                    break
            print(f"retorno total: {total:.2f}")
            return total

        try:
            run_episode(TreasureMazeEnv(), Q, max_steps=80, render=True)
        except NameError:
            print("treine o agente primeiro para criar a variável Q")
        """
    ),
    md(
        """
        ## Desafios para a turma

        1. **Leaderboard:** quem consegue maior retorno médio em 30 episódios gulosos?
        2. **Reward shaping:** mude as recompensas. O agente passa a pegar tesouros ou corre direto para a saída?
        3. **Ablation:** remova o vento `W`. O treinamento fica mais fácil?
        4. **Exploração:** compare `epsilon_end=0.30` com `epsilon_end=0.01`.
        5. **Algoritmo extra:** implemente SARSA e compare com Q-learning.
        6. **Rede neural:** treine uma rede para imitar sua tabela Q e discuta o que ainda falta para virar DQN.
        """
    ),
    code(
        """
        def evaluate(Q, episodes=30, max_steps=100):
            scores = []
            successes = 0
            for _ in range(episodes):
                env = TreasureMazeEnv()
                state = env.reset()
                total = 0.0
                done = False
                for _ in range(max_steps):
                    action = argmax_random_tie(Q[state])
                    state, reward, done, info = env.step(action)
                    total += reward
                    if done:
                        successes += 1
                        break
                scores.append(total)
            avg = sum(scores) / len(scores)
            print(f"retorno médio: {avg:.2f} | sucessos: {successes}/{episodes} | melhor: {max(scores):.2f}")
            return avg, successes

        try:
            evaluate(Q)
        except NameError:
            print("treine o agente primeiro para criar a variável Q")
        """
    ),
    md(
        """
        ## Gabarito do professor

        A célula abaixo substitui as funções TODO por uma implementação completa.

        Sugestão: deixe esta parte no seu notebook de professor e remova/oculte antes de distribuir aos alunos.
        """
    ),
    code(
        """
        def epsilon_greedy(Q, state, n_actions, epsilon):
            if random.random() < epsilon:
                return random.randrange(n_actions)
            return argmax_random_tie(Q[state])

        def q_learning_update(Q, state, action, reward, next_state, done, alpha, gamma):
            future = 0.0 if done else max(Q[next_state])
            target = reward + gamma * future
            Q[state][action] += alpha * (target - Q[state][action])
            return Q[state][action]

        def train_agent(env, episodes=3000, max_steps=160, alpha=0.20, gamma=0.96,
                        epsilon_start=1.0, epsilon_end=0.05, seed=SEED):
            random.seed(seed)
            Q = defaultdict(lambda: [0.0] * env.n_actions)
            returns = []

            for ep in range(episodes):
                frac = ep / max(1, episodes - 1)
                epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)
                state = env.reset()
                total = 0.0

                for _ in range(max_steps):
                    action = epsilon_greedy(Q, state, env.n_actions, epsilon)
                    next_state, reward, done, info = env.step(action)
                    q_learning_update(Q, state, action, reward, next_state, done, alpha, gamma)
                    state = next_state
                    total += reward
                    if done:
                        break

                returns.append(total)

            return Q, returns

        print("gabarito carregado")
        """
    ),
    code(
        """
        env = TreasureMazeEnv()
        Q, returns = train_agent(
            env,
            episodes=4500,
            max_steps=180,
            alpha=0.20,
            gamma=0.97,
            epsilon_start=1.0,
            epsilon_end=0.04,
        )
        plot_series(returns, "Labirinto do Tesouro - gabarito", ylabel="retorno", window=200)
        evaluate(Q)
        """
    ),
    md(
        """
        ## Extensão opcional - Rede neural e política

        Esta extensão é uma ponte para **Deep Q-Networks**.

        A rede abaixo recebe o estado do labirinto e tenta imitar a ação escolhida pela tabela Q:

        ```text
        rede_neural(estado) -> preferência por [cima, direita, baixo, esquerda]
        ```

        Aqui ela aprende imitando a política gulosa da tabela `Q` já treinada. Em um DQN completo, a rede preveria valores Q e aprenderia diretamente das transições `(estado, ação, recompensa, próximo_estado)`, usando replay buffer e uma rede alvo.
        """
    ),
    code(
        """
        class TinyMazePolicyNetwork:
            def __init__(self, n_inputs, n_hidden, n_outputs, lr=0.015, seed=SEED):
                random.seed(seed)
                self.lr = lr
                self.w1 = [[random.uniform(-0.35, 0.35) for _ in range(n_inputs)] for _ in range(n_hidden)]
                self.b1 = [0.0 for _ in range(n_hidden)]
                self.w2 = [[random.uniform(-0.35, 0.35) for _ in range(n_hidden)] for _ in range(n_outputs)]
                self.b2 = [0.0 for _ in range(n_outputs)]

            def forward(self, x):
                hidden = []
                for h, weights in enumerate(self.w1):
                    z = self.b1[h] + sum(w * xi for w, xi in zip(weights, x))
                    hidden.append(math.tanh(z))

                out = []
                for o, weights in enumerate(self.w2):
                    y = self.b2[o] + sum(w * hi for w, hi in zip(weights, hidden))
                    out.append(y)
                return hidden, out

            def predict(self, x):
                return self.forward(x)[1]

            def train_one(self, x, target):
                hidden, out = self.forward(x)
                grad_out = [2 * (out[o] - target[o]) / len(out) for o in range(len(out))]
                old_w2 = [row[:] for row in self.w2]

                for o in range(len(out)):
                    for h in range(len(hidden)):
                        self.w2[o][h] -= self.lr * grad_out[o] * hidden[h]
                    self.b2[o] -= self.lr * grad_out[o]

                for h in range(len(hidden)):
                    grad_h = sum(grad_out[o] * old_w2[o][h] for o in range(len(out)))
                    grad_h *= 1 - hidden[h] ** 2
                    for i in range(len(x)):
                        self.w1[h][i] -= self.lr * grad_h * x[i]
                    self.b1[h] -= self.lr * grad_h

                return sum((out[o] - target[o]) ** 2 for o in range(len(out))) / len(out)


        def encode_maze_state(state, env):
            row, col, has_key, treasure_mask = state
            features = [-1.0 for _ in range(env.height * env.width)]
            features[row * env.width + col] = 1.0
            features.append(has_key * 2 - 1)
            for idx in range(len(env.treasures)):
                features.append(1.0 if treasure_mask & (1 << idx) else -1.0)
            return features


        def greedy_actions_from_q_values(q_values):
            best = max(q_values)
            return [action for action, value in enumerate(q_values) if value == best]


        def train_neural_maze_policy(Q, epochs=90):
            env = TreasureMazeEnv()
            dataset = []
            for state, q_values in Q.items():
                if any(abs(value) > 1e-9 for value in q_values):
                    target = [-1.0 for _ in range(env.n_actions)]
                    best_actions = greedy_actions_from_q_values(q_values)
                    for action in best_actions:
                        target[action] = 1.0
                    dataset.append((encode_maze_state(state, env), target, best_actions))

            net = TinyMazePolicyNetwork(
                n_inputs=env.height * env.width + 1 + len(env.treasures),
                n_hidden=16,
                n_outputs=env.n_actions,
                lr=0.025,
            )
            losses = []
            for _ in range(epochs):
                random.shuffle(dataset)
                total_loss = 0.0
                for x, target, _ in dataset:
                    total_loss += net.train_one(x, target)
                losses.append(total_loss / max(1, len(dataset)))

            correct = 0
            for x, _, best_actions in dataset:
                predicted = argmax_random_tie(net.predict(x))
                if predicted in best_actions:
                    correct += 1

            print(f"amostras da tabela Q: {len(dataset)}")
            print(f"loss inicial: {losses[0]:.4f} | loss final: {losses[-1]:.4f}")
            print(f"acurácia imitando a ação gulosa: {correct / max(1, len(dataset)):.1%}")
            return net, losses


        def evaluate_neural_policy(net, episodes=30, max_steps=100):
            scores = []
            successes = 0
            for _ in range(episodes):
                env = TreasureMazeEnv()
                state = env.reset()
                total = 0.0
                for _ in range(max_steps):
                    values = net.predict(encode_maze_state(state, env))
                    action = argmax_random_tie(values)
                    state, reward, done, info = env.step(action)
                    total += reward
                    if done:
                        successes += 1
                        break
                scores.append(total)
            avg = sum(scores) / len(scores)
            print(f"rede neural | retorno médio: {avg:.2f} | sucessos: {successes}/{episodes} | melhor: {max(scores):.2f}")
            return avg, successes


        maze_net, nn_losses = train_neural_maze_policy(Q)
        evaluate_neural_policy(maze_net)
        plot_series(nn_losses, "Rede neural imitando a política Q no labirinto", ylabel="loss", window=5)
        """
    ),
])


outputs = {
    "01_aula1_rl_teoria_pratica_jogos.ipynb": aula1,
    "02_aula2_exercicio_labirinto_tesouro.ipynb": aula2,
}

for filename, nb in outputs.items():
    path = ROOT / filename
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"criado: {path}")
