import os
import time
import random
import numpy as np
import multiprocessing as mp
from dataclasses import dataclass
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

# ==========================================================
# CONFIG
# ==========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ajuste conforme seu hardware:
# - CPU fraca: 2-4
# - CPU mediana: 6-10
# - CPU forte: 12-32
NUM_ENVS = 8

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# ==========================================================
# REPLAY BUFFER (centralizado)
# - Thread-safe por lock (mesmo que o push aconteça no main)
# ==========================================================
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.buf = [None] * self.capacity
        self.pos = 0
        self.size = 0
        self.lock = mp.Lock()

    def push(self, s, a, r, ns, d):
        # Centralizado e protegido para suportar inserções concorrentes se necessário
        with self.lock:
            self.buf[self.pos] = (s, a, r, ns, d)
            self.pos = (self.pos + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        with self.lock:
            idx = np.random.randint(0, self.size, size=batch_size)
            batch = [self.buf[i] for i in idx]

        s, a, r, ns, d = map(np.stack, zip(*batch))
        return s, a, r, ns, d

    def __len__(self):
        return self.size


# ==========================================================
# DQN (mesma arquitetura base do seu código)
# ==========================================================
class DQN(nn.Module):
    def __init__(self, inp: int, out: int, h: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, out),
        )

    def forward(self, x):
        return self.net(x)


# ==========================================================
# ENV (copiado/compatível com o seu)
# ==========================================================
class LeaderFollowerEnv:
    def __init__(self):
        # Física
        self.dt = 0.05
        self.sub = 4
        self.delay = 4
        self.buf_size = 200

        # Ações
        self.v_vals = np.array([0.0, 0.25, 0.5])
        self.w_vals = np.array([-0.7, 0.0, 0.7])
        self.na = len(self.v_vals) * len(self.w_vals)

        # Seguidor
        self.max_v_f = 0.45
        self.kp_d = 1.0
        self.kp_a = 1.4

        # Mapa
        self.wall_x = 2.0
        self.door_min = -0.5
        self.door_max = 0.5
        self.wall_min = -3
        self.wall_max = 3

        # Limites
        self.xmin, self.xmax = -6, 6
        self.ymin, self.ymax = -6, 6

        # Spawn líder
        self.leader_start = np.array([self.wall_x - 4.0, 0.0, 0.0])

        # Spawn seguidor
        self.spawn_radius = 0.6

        # Zona de segurança
        self.min_safe_dist = 0.40
        self.collision_dist = 0.10

        # Distância desejada
        self.desired_dist = 0.30

        self.goal = None
        self.leader = np.zeros(3)
        self.follower = np.zeros(3)
        self.buffer = deque(maxlen=self.buf_size)

        # logging velocidades
        self.v_leader = 0.0
        self.w_leader = 0.0
        self.v_follower = 0.0
        self.w_follower = 0.0

    @staticmethod
    def wrap(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    def _wall(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        if (x1 - self.wall_x) * (x2 - self.wall_x) <= 0:
            if abs(x2 - x1) > 1e-9:
                t = (self.wall_x - x1) / (x2 - x1)
                if 0 <= t <= 1:
                    ycross = y1 + t * (y2 - y1)
                    if not (self.door_min <= ycross <= self.door_max):
                        return True
        return False

    def _step_uni(self, pose, v, w):
        x, y, th = pose
        for _ in range(self.sub):
            x += self.dt * v * np.cos(th)
            y += self.dt * v * np.sin(th)
            th = self.wrap(th + self.dt * w)
        return np.array([x, y, th])

    def _foll_ctrl(self, pose, tgt):
        x, y, th = pose
        tx, ty = tgt

        dx = tx - x
        dy = ty - y
        dist = np.hypot(dx, dy)

        ang = np.arctan2(dy, dx)
        err = self.wrap(ang - th)

        if dist < self.min_safe_dist:
            return 0.0, np.clip(self.kp_a * err, -1.0, 1.0)

        v = np.clip(self.kp_d * dist, 0, self.max_v_f)
        w = np.clip(self.kp_a * err, -1.5, 1.5)
        return v, w

    def _state(self):
        Lx, Ly, Lth = self.leader
        Fx, Fy, Fth = self.follower
        gx, gy = self.goal

        dist_goal = np.hypot(gx - Lx, gy - Ly)
        pair = np.hypot(Lx - Fx, Ly - Fy)
        heading = np.arctan2(gy - Ly, gx - Lx)
        ang = self.wrap(Lth - heading)

        return np.array(
            [
                gx - Lx,
                gy - Ly,
                dist_goal,
                pair,
                np.sin(Lth),
                np.cos(Lth),
                np.sin(Fth),
                np.cos(Fth),
                ang,
            ],
            dtype=np.float32,
        )

    def reset(self):
        self.leader = self.leader_start.copy()

        gx_min = self.wall_x + 2.5
        gx_max = min(self.wall_x + 4.5, self.xmax - 0.5)
        gx = np.random.uniform(gx_min, gx_max)
        gy = np.random.uniform(self.door_min + 0.1, self.door_max - 0.1)
        self.goal = np.array([gx, gy])

        for _ in range(40):
            ang = np.random.uniform(0, 2 * np.pi)
            d = np.random.uniform(0.1, self.spawn_radius)
            x2 = self.leader[0] + d * np.cos(ang)
            y2 = self.leader[1] + d * np.sin(ang)
            if not self._wall(self.leader[:2], np.array([x2, y2])):
                self.follower = np.array([x2, y2, self.leader[2]])
                break
        else:
            self.follower = self.leader.copy()

        self.buffer.clear()
        for _ in range(self.delay + 3):
            self.buffer.append(self.leader[:2].copy())

        st = self._state()
        self.last_goal = st[2]
        self.last_pair = st[3]
        self.last_ang = abs(st[8])
        return st

    def step(self, a: int):
        v = self.v_vals[a // 3]
        w = self.w_vals[a % 3]

        oldL = self.leader.copy()
        oldF = self.follower.copy()

        self.leader = self._step_uni(self.leader, v, w)
        self.buffer.append(self.leader[:2])

        tgt = self.buffer[-1 - self.delay]
        v2, w2 = self._foll_ctrl(self.follower, tgt)
        self.follower = self._step_uni(self.follower, v2, w2)

        self.v_leader = float(v)
        self.w_leader = float(w)
        self.v_follower = float(v2)
        self.w_follower = float(w2)

        for x, y in [(self.leader[0], self.leader[1]), (self.follower[0], self.follower[1])]:
            if x < self.xmin or x > self.xmax or y < self.ymin or y > self.ymax:
                return self._state(), -200.0, True, {"reason": "out"}

        if self._wall(oldL[:2], self.leader[:2]) or self._wall(oldF[:2], self.follower[:2]):
            return self._state(), -250.0, True, {"reason": "wall"}

        st = self._state()
        dist_goal = st[2]
        pair = st[3]
        ang = abs(st[8])

        reward = -0.05
        reward += 8 * (self.last_goal - dist_goal)
        reward += 2 * (abs(self.last_pair - self.desired_dist) - abs(pair - self.desired_dist))
        reward -= 0.3 * ang

        lat_error = abs(self.leader[1] - self.goal[1])
        reward -= 2.0 * lat_error

        if self.leader[1] < self.door_min - 0.3 or self.leader[1] > self.door_max + 0.3:
            reward -= 60

        if pair < self.min_safe_dist:
            reward -= 40

        if pair < self.collision_dist:
            return st, -150.0, True, {"reason": "robot_collision"}

        if oldL[0] < self.wall_x < self.leader[0]:
            if self.door_min <= self.leader[1] <= self.door_max:
                reward += 80

        if self.leader[0] > self.goal[0] + 0.1:
            reward += 180
            if abs(self.leader[1] - self.goal[1]) < 0.2:
                reward += 80
            else:
                reward -= 80

            if self.follower[0] < self.goal[0] < self.leader[0]:
                reward += 120
            else:
                reward -= 80

            return st, float(reward), True, {"reason": "passed_goal"}

        self.last_goal = dist_goal
        self.last_pair = pair
        self.last_ang = ang

        return st, float(reward), False, {}

    def poses(self):
        return self.leader.copy(), self.follower.copy(), self.goal.copy()


# ==========================================================
# WORKER PROCESS
# - Cada worker tem um env independente e só faz reset/step.
# - O paralelismo acontece aqui.
# ==========================================================
def env_worker(remote, seed: int):
    try:
        np.random.seed(seed)
        random.seed(seed)

        env = LeaderFollowerEnv()
        obs = env.reset()

        remote.send(("reset_ok", obs))

        while True:
            cmd, data = remote.recv()

            if cmd == "step":
                action = int(data)
                next_obs, reward, done, info = env.step(action)
                if done:
                    next_obs = env.reset()  # auto-reset local
                remote.send(("step_ok", (next_obs, reward, done, info)))

            elif cmd == "get_obs":
                remote.send(("obs_ok", obs))

            elif cmd == "close":
                remote.close()
                break

            else:
                remote.send(("err", f"Unknown cmd: {cmd}"))

            # guarda obs atual (para depuração se quiser)
            if cmd == "step":
                obs = next_obs

    except Exception as e:
        remote.send(("crash", repr(e)))
        remote.close()


class ParallelEnvs:
    """
    Gerenciador simples tipo SubprocVectorEnv, mas independente de Gym.
    - Uma ação por env
    - step() retorna batch (obs, rewards, dones, infos)
    """

    def __init__(self, n_envs: int, base_seed: int = 0):
        self.n_envs = int(n_envs)
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.n_envs)])
        self.ps = []

        for i in range(self.n_envs):
            s = base_seed + 10_000 * i
            p = mp.Process(target=env_worker, args=(self.work_remotes[i], s), daemon=True)
            p.start()
            self.ps.append(p)
            self.work_remotes[i].close()

        # receber reset de todos
        obs = []
        for r in self.remotes:
            msg, payload = r.recv()
            assert msg == "reset_ok", (msg, payload)
            obs.append(payload)

        self.obs = np.stack(obs, axis=0)  # (n_envs, obs_dim)

    def step(self, actions: np.ndarray):
        """
        Paralelismo:
        - envia ações para todos os workers
        - recebe respostas de todos
        """
        assert len(actions) == self.n_envs

        for r, a in zip(self.remotes, actions):
            r.send(("step", int(a)))

        next_obs, rewards, dones, infos = [], [], [], []
        for r in self.remotes:
            msg, payload = r.recv()
            if msg == "crash":
                raise RuntimeError(f"Worker crashed: {payload}")
            assert msg == "step_ok", (msg, payload)

            ob, rew, done, info = payload
            next_obs.append(ob)
            rewards.append(rew)
            dones.append(done)
            infos.append(info)

        self.obs = np.stack(next_obs, axis=0)
        return self.obs, np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.bool_), infos

    def close(self):
        for r in self.remotes:
            try:
                r.send(("close", None))
            except Exception:
                pass
        for p in self.ps:
            p.join(timeout=1.0)


# ==========================================================
# TRAINER (DQN único + buffer central)
# ==========================================================
@dataclass
class TrainConfig:
    # ===== RL =====
    gamma: float = 0.99
    batch: int = 64
    lr: float = 1e-3

    # ===== Replay Buffer =====
    buffer_capacity: int = 100000
    min_buffer: int = 2000

    # ===== Epsilon =====
    eps_start: float = 1.0
    eps_min: float = 0.05
    eps_decay: float = 0.9993

    # ===== Paralelismo =====
    num_envs: int = 8

    # ===== Treino =====
    max_episodes: int = 2000      # 🔹 AGORA CONFIGURÁVEL
    train_every: int = 1
    target_sync_every: int = 1500
    grad_steps_per_update: int = 1

    print_every_episodes: int = 200
    log_transitions: bool = False   

class DQNParallelTrainer:
    def __init__(self, n_envs: int, cfg: TrainConfig):
        self.cfg = cfg
        self.envs = ParallelEnvs(n_envs=n_envs, base_seed=SEED)
       

        obs_dim = self.envs.obs.shape[1]
        # ação vem do próprio env
        tmp_env = LeaderFollowerEnv()
        tmp_env.reset()
        act_dim = tmp_env.na

        self.policy = DQN(obs_dim, act_dim).to(DEVICE)
        self.target = DQN(obs_dim, act_dim).to(DEVICE)
        self.target.load_state_dict(self.policy.state_dict())

        self.opt = optim.Adam(self.policy.parameters(), lr=cfg.lr)
        self.buf = ReplayBuffer(cfg.buffer_capacity)

        self.eps = cfg.eps_start
        self.global_step = 0  # conta steps "vetorizados"? aqui contamos steps globais por transição
        self.total_transitions = 0

        self.episode_count = 0
        self.last_logged_episode = -1
        self.episode_rewards = np.zeros(self.envs.n_envs)

    @torch.no_grad()
    def act_batch(self, obs_batch: np.ndarray):
        """
        Seleciona ações para todos os envs:
        - eps-greedy
        - UMA rede para todos
        """
        n = obs_batch.shape[0]
        actions = np.zeros(n, dtype=np.int64)

        # amostras aleatórias
        rand_mask = np.random.rand(n) < self.eps
        actions[rand_mask] = np.random.randint(0, self.policy.net[-1].out_features, size=rand_mask.sum())

        # ações greedy
        if (~rand_mask).any():
            obs_t = torch.from_numpy(obs_batch[~rand_mask]).float().to(DEVICE)
            q = self.policy(obs_t)
            actions[~rand_mask] = q.argmax(dim=1).cpu().numpy().astype(np.int64)

        return actions

    def update(self):
        if len(self.buf) < self.cfg.min_buffer:
            return

        for _ in range(self.cfg.grad_steps_per_update):
            s, a, r, ns, d = self.buf.sample(self.cfg.batch)

            s = torch.from_numpy(s).float().to(DEVICE)
            ns = torch.from_numpy(ns).float().to(DEVICE)
            a = torch.from_numpy(a).long().to(DEVICE).view(-1, 1)
            r = torch.from_numpy(r).float().to(DEVICE).view(-1, 1)
            d = torch.from_numpy(d).float().to(DEVICE).view(-1, 1)

            q = self.policy(s).gather(1, a)
            with torch.no_grad():
                max_next = self.target(ns).max(dim=1, keepdim=True)[0]
                target = r + (1.0 - d) * self.cfg.gamma * max_next

            loss = nn.MSELoss()(q, target)

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()

    def train(self):
        start_time = time.time()
        obs = self.envs.obs
        t0 = time.time()

        while self.episode_count < self.cfg.max_episodes:
            # 1) ações
            actions = self.act_batch(obs)

            # 2) step paralelo
            next_obs, rewards, dones, infos = self.envs.step(actions)

            for i in range(self.envs.n_envs):
                # replay buffer (UMA VEZ)
                self.buf.push(
                    obs[i],
                    actions[i],
                    rewards[i],
                    next_obs[i],
                    float(dones[i]),
                )

                self.episode_rewards[i] += rewards[i]

                # episódio terminou
                if dones[i]:
                    self.episode_count += 1
                    self.episode_rewards[i] = 0.0

                    # STATUS controlado
                    if (
                        self.episode_count > 0
                        and self.episode_count % self.cfg.print_every_episodes == 0
                        and self.episode_count != self.last_logged_episode
                    ):
                        print(
                            f"[STATUS] Episodes={self.episode_count}/{self.cfg.max_episodes} | "
                            f"Eps={self.eps:.3f} | Buffer={len(self.buf)}"
                        )
                        self.last_logged_episode = self.episode_count

            obs = next_obs
            self.total_transitions += self.envs.n_envs

            # 3) treino
            if (self.total_transitions // self.envs.n_envs) % self.cfg.train_every == 0:
                self.update()

            # 4) sync target
            if (self.total_transitions // self.envs.n_envs) % self.cfg.target_sync_every == 0:
                self.target.load_state_dict(self.policy.state_dict())

            # 5) epsilon decay
            self.eps = max(self.cfg.eps_min, self.eps * self.cfg.eps_decay)

            # log de throughput (independente)
            if self.cfg.log_transitions:
                if self.total_transitions % (self.envs.n_envs * 500) == 0:
                    dt = time.time() - t0
                    sps = self.total_transitions / max(dt, 1e-6)
                    print(
                        f"[train] transitions={self.total_transitions} | "
                        f"eps={self.eps:.3f} | buffer={len(self.buf)} | SPS={sps:.0f}"
                    )


        # tempo final
        elapsed = time.time() - start_time
        avg_time_per_ep = elapsed / max(1, self.episode_count)

        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = elapsed % 60

        print("\n================ TREINO FINALIZADO ================")
        print(f"Total de episódios: {self.episode_count}")
        print(f"Tempo total de treino: {h:02d}:{m:02d}:{s:05.2f}")
        print(f"Tempo médio por episódio: {avg_time_per_ep:.3f} s")
        print("===================================================\n")

        self.envs.close()



# ==========================================================
# COLETA DE UM EPISÓDIO (CSV)
# ==========================================================
import csv
import matplotlib.pyplot as plt
import pandas as pd


def run_episode_collect(agent, set_start_fn, episode_id, out_dir="logs", max_steps=300):
    os.makedirs(out_dir, exist_ok=True)

    env = agent.env if hasattr(agent, "env") else LeaderFollowerEnv()
    st = env.reset()
    set_start_fn(env)
    st = env._state()

    plot_initial_positions(env, title=f"Cenário inicial – {episode_id.upper()}")

    filepath = f"{out_dir}/episode_{episode_id}.csv"

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "t",
            "leader_x","leader_y","leader_theta",
            "leader_v","leader_w",
            "follower_x","follower_y","follower_theta",
            "follower_v","follower_w",
            "reward","reason"
        ])

        t = 0.0
        done = False
        steps = 0
        reason = ""

        l = env.leader
        f0 = env.follower
        writer.writerow([
            t,
            l[0], l[1], l[2],
            env.v_leader, env.w_leader,
            f0[0], f0[1], f0[2],
            env.v_follower, env.w_follower,
            0.0, ""
        ])

        while not done and steps < max_steps:
            a = agent.act(st, greedy=True)
            st, r, done, info = env.step(a)

            t += env.dt * env.sub
            steps += 1
            l = env.leader
            f0 = env.follower
            reason = info.get("reason", "")

            writer.writerow([
                t,
                l[0], l[1], l[2],
                env.v_leader, env.w_leader,
                f0[0], f0[1], f0[2],
                env.v_follower, env.w_follower,
                r,
                reason if done else ""
            ])

    return filepath


# ==========================================================
# PLOTS DE UM EPISÓDIO
# ==========================================================
def plot_episode(filepath, out_dir=None):
    df = pd.read_csv(filepath)
    df = df[df["leader_x"].notna()]
    t = df["t"]

    if out_dir is None:
        out_dir = os.path.dirname(filepath)
    os.makedirs(out_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(filepath))[0]

    # === POSIÇÃO ===
    plt.figure(figsize=(10, 6))
    plt.plot(t, df["leader_x"], label="Leader x")
    plt.plot(t, df["leader_y"], label="Leader y")
    plt.plot(t, df["follower_x"], label="Follower x")
    plt.plot(t, df["follower_y"], label="Follower y")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Posição (m)")
    plt.title("Posição vs Tempo")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, f"{base}_pose.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # === VELOCIDADES ===
    plt.figure(figsize=(10, 6))
    plt.plot(t, df["leader_v"], label="Leader v")
    plt.plot(t, df["leader_w"], label="Leader w")
    plt.plot(t, df["follower_v"], label="Follower v")
    plt.plot(t, df["follower_w"], label="Follower w")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Velocidade")
    plt.title("Velocidades vs Tempo")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, f"{base}_vel.png"), dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[OK] Gráficos salvos para {base}")


# ==========================================================
# PLOT DAS POSIÇÕES INICIAIS
# ==========================================================
def plot_initial_positions(env, title="Posição inicial"):
    plt.figure(figsize=(6, 6))
    plt.xlim(env.xmin, env.xmax)
    plt.ylim(env.ymin, env.ymax)
    plt.gca().set_aspect("equal")

    xw = env.wall_x
    plt.plot([xw, xw], [env.wall_min, env.door_min], "k-", lw=4)
    plt.plot([xw, xw], [env.door_max, env.wall_max], "k-", lw=4)
    plt.plot([xw, xw], [env.door_min, env.door_max], "g-", lw=6)

    plt.scatter(env.leader[0], env.leader[1], c="blue", s=80, label="Leader")
    plt.scatter(env.follower[0], env.follower[1], c="red", s=80, label="Follower")
    plt.scatter(env.goal[0], env.goal[1], c="green", s=120, marker="*", label="Goal")

    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# ==========================================================
# DEFINIÇÃO DOS CENÁRIOS (EASY / MEDIUM / HARD)
# ==========================================================
def set_start_easy(env):
    """
    Cenário fácil:
    - líder alinhado
    - seguidor logo atrás
    """
    env.leader = env.leader_start.copy()
    env.follower = env.leader.copy()
    env.follower[0] -= 0.4
    env.follower[2] = env.leader[2]


def set_start_medium(env, angle_deg=45, distance=1.2):
    """
    Cenário médio:
    - líder com rotação
    - deslocamento lateral moderado
    """
    theta = np.deg2rad(angle_deg)
    env.leader = env.leader_start.copy()
    env.leader[2] = theta
    env.leader[0] -= distance * np.cos(theta)
    env.leader[1] -= distance * np.sin(theta)

    env.follower = env.leader.copy()
    env.follower[0] -= 0.6 * np.cos(theta)
    env.follower[1] -= 0.6 * np.sin(theta)


def set_start_hard(env, angle_deg=90, distance=1.2):
    """
    Cenário difícil:
    - líder perpendicular à porta
    - erro angular máximo
    """
    theta = np.deg2rad(angle_deg)
    env.leader = env.leader_start.copy()
    env.leader[2] = theta
    env.leader[0] -= distance * np.cos(theta)
    env.leader[1] -= distance * np.sin(theta)

    env.follower = env.leader.copy()
    env.follower[0] -= 0.6 * np.cos(theta)
    env.follower[1] -= 0.6 * np.sin(theta)

# ==========================================================
# AGENTE DE AVALIAÇÃO (WRAPPER)
# ==========================================================
class EvalAgent:
    """
    Wrapper leve para avaliação:
    - usa a policy treinada no DQN paralelo
    - mantém interface compatível com o código original
    """
    def __init__(self, policy, env):
        self.policy = policy
        self.env = env

    def act(self, state, greedy=True):
        with torch.no_grad():
            s = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
            q = self.policy(s)
            return int(q.argmax())
# ==========================================================
# AVALIAÇÃO POR DIFICULDADE (EASY / MEDIUM / HARD)
# ==========================================================
def evaluate_difficulty(agent, set_start_fn, name, n_episodes=10):
    successes = 0
    reasons = {}

    for i in range(n_episodes):
        env = agent.env
        st = env.reset()
        set_start_fn(env)
        st = env._state()

        done = False
        steps = 0
        info = {}

        while not done and steps < 300:
            a = agent.act(st, greedy=True)
            st, r, done, info = env.step(a)
            steps += 1

        reason = info.get("reason", "unknown")
        reasons[reason] = reasons.get(reason, 0) + 1

        if reason == "passed_goal":
            successes += 1

    success_rate = successes / n_episodes

    # ===== PRINT CLARO (ARTIGO / TERMINAL) =====
    print("\n==============================")
    print(f"Dificuldade: {name.upper()}")
    print(f"Sucessos: {successes}/{n_episodes}")
    print(f"Taxa de sucesso: {success_rate:.2f}")
    print("Motivos de término:")
    for k, v in reasons.items():
        print(f"  {k}: {v}")
    print("==============================\n")

    return success_rate, reasons


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    # =========================
    # CONFIGURAÇÃO DE TREINO
    # =========================
    cfg = TrainConfig(
        gamma=0.99,
        batch=64,
        lr=1e-3,
        buffer_capacity=100000,
        min_buffer=2000,
        eps_start=1.0,
        eps_min=0.05,
        eps_decay=0.9993,
        max_episodes=2000,
        train_every=1,
        target_sync_every=1500,
        grad_steps_per_update=1,
    )

    print(f"Rodando com NUM_ENVS={NUM_ENVS}")
    trainer = DQNParallelTrainer(n_envs=NUM_ENVS, cfg=cfg)

    # =========================
    # TREINO
    # =========================
    trainer.train()

    # =========================
    # AGENTE DE AVALIAÇÃO
    # =========================
    eval_env = LeaderFollowerEnv()
    agent = EvalAgent(trainer.policy, eval_env)

    # =========================
    # CENÁRIOS DE TESTE
    # =========================
    scenarios = [
        ("easy", set_start_easy),
        ("medium", set_start_medium),
        ("hard", set_start_hard),
    ]

    # =========================
    # COLETA + GRÁFICOS
    # =========================
    for name, start_fn in scenarios:
        print(f"\n=== COLETA | {name.upper()} ===")

        csv_path = run_episode_collect(
            agent,
            start_fn,
            episode_id=name,
            out_dir=f"logs/{name}"
        )

        plot_episode(csv_path)

    # =========================
    # AVALIAÇÃO ESTATÍSTICA
    # =========================
    for name, fn in scenarios:
        evaluate_difficulty(agent, fn, name, n_episodes=500)

    print("\nExecução finalizada com sucesso.")
