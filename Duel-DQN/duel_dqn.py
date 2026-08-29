import os
import time
import random
import csv
from dataclasses import dataclass
from collections import deque
import multiprocessing as mp

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

# =====================
# CONFIG
# =====================
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


# =====================
# REPLAY BUFFER (centralizado)
# - Thread-safe por lock
# =====================
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.buf = [None] * self.capacity
        self.pos = 0
        self.size = 0
        self.lock = mp.Lock()

    def push(self, s, a, r, ns, d):
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


# =====================
# DUELING DQN (mais profundo)
# =====================
class DeepDuelingDQN(nn.Module):
    """Dueling DQN:
    Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))
    """

    def __init__(self, inp: int, out: int, h: int = 512, depth: int = 3, dropout: float = 0.0):
        super().__init__()

        layers = []
        last = inp
        for i in range(depth):
            layers.append(nn.Linear(last, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(p=float(dropout)))
            last = h
        self.feature = nn.Sequential(*layers)

        self.value = nn.Sequential(
            nn.Linear(h, h // 2),
            nn.ReLU(),
            nn.Linear(h // 2, 1),
        )

        self.adv = nn.Sequential(
            nn.Linear(h, h // 2),
            nn.ReLU(),
            nn.Linear(h // 2, out),
        )

    def forward(self, x):
        z = self.feature(x)
        v = self.value(z)
        a = self.adv(z)
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q


# =====================
# ENV: Sala fechada + brecha + porta + spawn amarelo
# =====================
class LeaderFollowerEnv:
    def __init__(self):
        # Física
        self.dt = 0.05
        self.sub = 4
        self.delay = 4
        self.buf_size = 200

        # Ações discretas
        self.v_vals = np.array([0.0, 0.25, 0.5], dtype=np.float32)
        self.w_vals = np.array([-0.7, 0.0, 0.7], dtype=np.float32)
        self.na = int(len(self.v_vals) * len(self.w_vals))

        # Seguidor (controle proporcional simples)
        self.max_v_f = 0.45
        self.kp_d = 1.0
        self.kp_a = 1.4

        # Limites globais (apenas segurança numérica)
        self.gx_min, self.gx_max = -8.0, 6.0
        self.gy_min, self.gy_max = -7.0, 5.0

        # =====================
        # Geometria da sala (paredes externas)
        # =====================
        self.room_xmin, self.room_xmax = -7.0, 1.0
        self.room_ymin, self.room_ymax = -6.0, 4.0

        # Porta lateral (na parede da direita da sala)
        self.door_wall_x = self.room_xmax
        self.door_y_min = -1.5
        self.door_y_max = 0.5

        # Parede interna horizontal com brecha (gap)
        self.gap_wall_y = -1.0
        self.gap_wall_x0 = self.room_xmin
        self.gap_wall_x1 = -1.0  # termina antes do lado direito
        self.gap_x_min = -3.0
        self.gap_x_max = -2.0

        # Spawn amarelo (retângulo)
        self.spawn_xmin, self.spawn_xmax = -6.5, -4.5
        self.spawn_ymin, self.spawn_ymax = -5.5, -3.5

        # Distâncias
        self.min_safe_dist = 0.40
        self.collision_dist = 0.10
        self.desired_dist = 0.30
        self.spawn_radius = 0.6

        # Estados
        self.goal = None
        self.leader = np.zeros(3, dtype=np.float32)
        self.follower = np.zeros(3, dtype=np.float32)
        self.buffer = deque(maxlen=self.buf_size)

        # logging velocidades
        self.v_leader = 0.0
        self.w_leader = 0.0
        self.v_follower = 0.0
        self.w_follower = 0.0

        # flags de checkpoints
        self.passed_gap = False
        self.passed_door = False

        # memória para shaping
        self.last_goal = 0.0
        self.last_pair = 0.0
        self.last_ang = 0.0

        # segmentos de parede (para colisão por interseção de segmento)
        self._build_walls()

    @staticmethod
    def wrap(a: float) -> float:
        return (a + np.pi) % (2 * np.pi) - np.pi

    def _build_walls(self):
        """Define paredes como lista de segmentos (p1, p2)."""
        xmin, xmax, ymin, ymax = self.room_xmin, self.room_xmax, self.room_ymin, self.room_ymax

        walls = []

        # Parede esquerda (x=xmin)
        walls.append(((xmin, ymin), (xmin, ymax)))

        # Parede superior (y=ymax)
        walls.append(((xmin, ymax), (xmax, ymax)))

        # Parede inferior (y=ymin)
        walls.append(((xmin, ymin), (xmax, ymin)))

        # Parede direita (x=xmax) com abertura (porta)
        walls.append(((xmax, ymin), (xmax, self.door_y_min)))
        walls.append(((xmax, self.door_y_max), (xmax, ymax)))

        # Parede interna horizontal (y=gap_wall_y) com brecha
        y = self.gap_wall_y
        x0, x1 = self.gap_wall_x0, self.gap_wall_x1
        gx0, gx1 = self.gap_x_min, self.gap_x_max
        walls.append(((x0, y), (gx0, y)))
        walls.append(((gx1, y), (x1, y)))

        self.walls = walls

    def _segment_intersect(self, p1, p2, q1, q2) -> bool:
        """Retorna True se os segmentos p1-p2 e q1-q2 se intersectam."""
        def orient(a, b, c):
            return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

        def on_segment(a, b, c):
            # c colinear com a-b e dentro do retângulo
            return (
                min(a[0], b[0]) - 1e-9 <= c[0] <= max(a[0], b[0]) + 1e-9
                and min(a[1], b[1]) - 1e-9 <= c[1] <= max(a[1], b[1]) + 1e-9
            )

        o1 = orient(p1, p2, q1)
        o2 = orient(p1, p2, q2)
        o3 = orient(q1, q2, p1)
        o4 = orient(q1, q2, p2)

        # geral
        if (o1 * o2 < 0) and (o3 * o4 < 0):
            return True

        # casos colineares
        if abs(o1) < 1e-9 and on_segment(p1, p2, q1):
            return True
        if abs(o2) < 1e-9 and on_segment(p1, p2, q2):
            return True
        if abs(o3) < 1e-9 and on_segment(q1, q2, p1):
            return True
        if abs(o4) < 1e-9 and on_segment(q1, q2, p2):
            return True

        return False

    def _hits_any_wall(self, old_xy, new_xy) -> bool:
        p1 = (float(old_xy[0]), float(old_xy[1]))
        p2 = (float(new_xy[0]), float(new_xy[1]))
        for (q1, q2) in self.walls:
            if self._segment_intersect(p1, p2, q1, q2):
                return True
        return False

    def _out_of_global_bounds(self, pose) -> bool:
        x, y = float(pose[0]), float(pose[1])
        return (x < self.gx_min) or (x > self.gx_max) or (y < self.gy_min) or (y > self.gy_max)

    def _inside_spawn(self, x, y) -> bool:
        return (
            self.spawn_xmin <= x <= self.spawn_xmax
            and self.spawn_ymin <= y <= self.spawn_ymax
        )

    def _step_uni(self, pose, v, w):
        x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
        for _ in range(self.sub):
            x += self.dt * float(v) * np.cos(th)
            y += self.dt * float(v) * np.sin(th)
            th = self.wrap(th + self.dt * float(w))
        return np.array([x, y, th], dtype=np.float32)

    def _foll_ctrl(self, pose, tgt):
        x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
        tx, ty = float(tgt[0]), float(tgt[1])

        dx = tx - x
        dy = ty - y
        dist = float(np.hypot(dx, dy))

        ang = float(np.arctan2(dy, dx))
        err = self.wrap(ang - th)

        if dist < self.min_safe_dist:
            return 0.0, float(np.clip(self.kp_a * err, -1.0, 1.0))

        v = float(np.clip(self.kp_d * dist, 0.0, self.max_v_f))
        w = float(np.clip(self.kp_a * err, -1.5, 1.5))
        return v, w

    def _state(self):
        Lx, Ly, Lth = self.leader
        Fx, Fy, Fth = self.follower
        gx, gy = self.goal

        dist_goal = float(np.hypot(gx - Lx, gy - Ly))
        pair = float(np.hypot(Lx - Fx, Ly - Fy))
        heading = float(np.arctan2(gy - Ly, gx - Lx))
        ang = float(self.wrap(Lth - heading))

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
        # Spawn do líder dentro da zona amarela
        for _ in range(100):
            lx = float(np.random.uniform(self.spawn_xmin, self.spawn_xmax))
            ly = float(np.random.uniform(self.spawn_ymin, self.spawn_ymax))
            if self._inside_spawn(lx, ly):
                break
        lth = float(np.random.uniform(-np.pi, np.pi))
        self.leader = np.array([lx, ly, lth], dtype=np.float32)

        # Spawn do seguidor próximo ao líder, também na zona amarela
        placed = False
        for _ in range(60):
            ang = float(np.random.uniform(0, 2 * np.pi))
            d = float(np.random.uniform(0.1, self.spawn_radius))
            x2 = lx + d * np.cos(ang)
            y2 = ly + d * np.sin(ang)
            if self._inside_spawn(x2, y2):
                self.follower = np.array([x2, y2, lth], dtype=np.float32)
                placed = True
                break
        if not placed:
            self.follower = np.array([lx - 0.3, ly, lth], dtype=np.float32)

        # Goal: além da porta (direita), alinhado com faixa da porta
        gx = float(np.random.uniform(self.room_xmax + 2.5, min(self.room_xmax + 4.5, self.gx_max - 0.2)))
        gy = float(np.random.uniform(self.door_y_min + 0.2, self.door_y_max - 0.2))
        self.goal = np.array([gx, gy], dtype=np.float32)

        self.buffer.clear()
        for _ in range(self.delay + 3):
            self.buffer.append(self.leader[:2].copy())

        # Reset checkpoints
        self.passed_gap = False
        self.passed_door = False

        st = self._state()
        self.last_goal = float(st[2])
        self.last_pair = float(st[3])
        self.last_ang = float(abs(st[8]))
        return st

    def _crossed_line_y(self, old_xy, new_xy, y_line: float) -> bool:
        y1 = float(old_xy[1])
        y2 = float(new_xy[1])
        return (y1 - y_line) * (y2 - y_line) <= 0.0

    def _crossed_line_x(self, old_xy, new_xy, x_line: float) -> bool:
        x1 = float(old_xy[0])
        x2 = float(new_xy[0])
        return (x1 - x_line) * (x2 - x_line) <= 0.0

    def _interp_x_at_y(self, old_xy, new_xy, y_line: float) -> float:
        x1, y1 = float(old_xy[0]), float(old_xy[1])
        x2, y2 = float(new_xy[0]), float(new_xy[1])
        if abs(y2 - y1) < 1e-9:
            return x2
        t = (y_line - y1) / (y2 - y1)
        return x1 + t * (x2 - x1)

    def _interp_y_at_x(self, old_xy, new_xy, x_line: float) -> float:
        x1, y1 = float(old_xy[0]), float(old_xy[1])
        x2, y2 = float(new_xy[0]), float(new_xy[1])
        if abs(x2 - x1) < 1e-9:
            return y2
        t = (x_line - x1) / (x2 - x1)
        return y1 + t * (y2 - y1)

    def step(self, a: int):
        v = float(self.v_vals[a // 3])
        w = float(self.w_vals[a % 3])

        oldL = self.leader.copy()
        oldF = self.follower.copy()

        # líder
        self.leader = self._step_uni(self.leader, v, w)
        self.buffer.append(self.leader[:2])

        # seguidor
        tgt = self.buffer[-1 - self.delay]
        v2, w2 = self._foll_ctrl(self.follower, tgt)
        self.follower = self._step_uni(self.follower, v2, w2)

        self.v_leader = v
        self.w_leader = w
        self.v_follower = float(v2)
        self.w_follower = float(w2)

        # segurança global
        if self._out_of_global_bounds(self.leader) or self._out_of_global_bounds(self.follower):
            return self._state(), -200.0, True, {"reason": "out"}

        # colisão com paredes
        if self._hits_any_wall(oldL[:2], self.leader[:2]) or self._hits_any_wall(oldF[:2], self.follower[:2]):
            return self._state(), -250.0, True, {"reason": "wall"}

        st = self._state()
        dist_goal = float(st[2])
        pair = float(st[3])
        ang = float(abs(st[8]))

        # shaping base
        reward = -0.05
        reward += 8.0 * (self.last_goal - dist_goal)
        reward += 2.0 * (abs(self.last_pair - self.desired_dist) - abs(pair - self.desired_dist))
        reward -= 0.3 * ang

        # penalidade por afastar da faixa da porta (mantém navegação "reta")
        lat_error = float(abs(self.leader[1] - self.goal[1]))
        reward -= 1.0 * lat_error

        # segurança entre robôs
        if pair < self.min_safe_dist:
            reward -= 25.0
        if pair < self.collision_dist:
            return st, -150.0, True, {"reason": "robot_collision"}

        # =====================
        # CHECKPOINT 1: brecha (gap) na parede interna
        # =====================
        if not self.passed_gap:
            if self._crossed_line_y(oldL[:2], self.leader[:2], self.gap_wall_y):
                xcross = self._interp_x_at_y(oldL[:2], self.leader[:2], self.gap_wall_y)
                if self.gap_x_min <= xcross <= self.gap_x_max:
                    reward += 120.0
                    self.passed_gap = True

        # =====================
        # CHECKPOINT 2: porta
        # =====================
        if self.passed_gap and (not self.passed_door):
            if self._crossed_line_x(oldL[:2], self.leader[:2], self.door_wall_x):
                ycross = self._interp_y_at_x(oldL[:2], self.leader[:2], self.door_wall_x)
                if self.door_y_min <= ycross <= self.door_y_max:
                    reward += 200.0
                    self.passed_door = True

                    # bônus se seguidor já cruzou a porta também
                    if self.follower[0] >= self.door_wall_x:
                        reward += 120.0
                    else:
                        reward -= 60.0

        # término ao passar o goal (após porta)
        if self.passed_door and (self.leader[0] > self.goal[0] + 0.1):
            reward += 180.0
            if abs(self.leader[1] - self.goal[1]) < 0.35:
                reward += 80.0
            else:
                reward -= 60.0

            return st, float(reward), True, {"reason": "passed_goal"}

        # atualiza memória de shaping
        self.last_goal = dist_goal
        self.last_pair = pair
        self.last_ang = ang

        return st, float(reward), False, {}

    def poses(self):
        return self.leader.copy(), self.follower.copy(), self.goal.copy()


# =====================
# WORKER PROCESS (envs em paralelo)
# =====================
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
                    next_obs = env.reset()
                remote.send(("step_ok", (next_obs, reward, done, info)))

                obs = next_obs

            elif cmd == "close":
                remote.close()
                break

            else:
                remote.send(("err", f"Unknown cmd: {cmd}"))

    except Exception as e:
        try:
            remote.send(("crash", repr(e)))
        except Exception:
            pass
        remote.close()


class ParallelEnvs:
    """Gerenciador simples tipo SubprocVectorEnv, sem Gym."""

    def __init__(self, n_envs: int, base_seed: int = 0):
        self.n_envs = int(n_envs)
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.n_envs)])
        self.ps = []

        for i in range(self.n_envs):
            s = int(base_seed + 10_000 * i)
            p = mp.Process(target=env_worker, args=(self.work_remotes[i], s), daemon=True)
            p.start()
            self.ps.append(p)
            self.work_remotes[i].close()

        obs = []
        for r in self.remotes:
            msg, payload = r.recv()
            if msg != "reset_ok":
                raise RuntimeError((msg, payload))
            obs.append(payload)

        self.obs = np.stack(obs, axis=0)

    def step(self, actions: np.ndarray):
        assert len(actions) == self.n_envs

        for r, a in zip(self.remotes, actions):
            r.send(("step", int(a)))

        next_obs, rewards, dones, infos = [], [], [], []
        for r in self.remotes:
            msg, payload = r.recv()
            if msg == "crash":
                raise RuntimeError(f"Worker crashed: {payload}")
            if msg != "step_ok":
                raise RuntimeError((msg, payload))

            ob, rew, done, info = payload
            next_obs.append(ob)
            rewards.append(rew)
            dones.append(done)
            infos.append(info)

        self.obs = np.stack(next_obs, axis=0)
        return (
            self.obs,
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.bool_),
            infos,
        )

    def close(self):
        for r in self.remotes:
            try:
                r.send(("close", None))
            except Exception:
                pass
        for p in self.ps:
            p.join(timeout=1.0)


# =====================
# TRAINER
# =====================
@dataclass
class TrainConfig:
    # RL
    gamma: float = 0.99
    batch: int = 128
    lr: float = 1e-3

    # Buffer
    buffer_capacity: int = 30000
    min_buffer: int = 10000

    # Epsilon
    eps_start: float = 1.0
    eps_min: float = 0.02
    eps_decay: float = 0.9993

    # Paralelismo
    num_envs: int = 8

    # Treino
    max_episodes: int = 10000
    train_every: int = 1
    target_sync_every: int = 1500
    grad_steps_per_update: int = 4

    # Logs
    print_every_episodes: int = 200
    log_transitions: bool = False

    # Modelo
    hidden: int = 512
    depth: int = 3
    dropout: float = 0.0


class DQNParallelTrainer:
    def __init__(self, n_envs: int, cfg: TrainConfig):
        self.cfg = cfg
        self.envs = ParallelEnvs(n_envs=n_envs, base_seed=SEED)

        obs_dim = int(self.envs.obs.shape[1])

        # ação vem do env
        tmp_env = LeaderFollowerEnv()
        tmp_env.reset()
        act_dim = int(tmp_env.na)

        self.policy = DeepDuelingDQN(obs_dim, act_dim, h=cfg.hidden, depth=cfg.depth, dropout=cfg.dropout).to(DEVICE)
        self.target = DeepDuelingDQN(obs_dim, act_dim, h=cfg.hidden, depth=cfg.depth, dropout=cfg.dropout).to(DEVICE)
        self.target.load_state_dict(self.policy.state_dict())

        self.opt = optim.Adam(self.policy.parameters(), lr=cfg.lr)
        self.buf = ReplayBuffer(cfg.buffer_capacity)

        self.eps = float(cfg.eps_start)
        self.total_transitions = 0

        self.episode_count = 0
        self.last_logged_episode = -1
        self.episode_rewards = np.zeros(self.envs.n_envs, dtype=np.float32)

    @torch.no_grad()
    def act_batch(self, obs_batch: np.ndarray):
        n = int(obs_batch.shape[0])
        actions = np.zeros(n, dtype=np.int64)

        rand_mask = np.random.rand(n) < self.eps
        if rand_mask.any():
            actions[rand_mask] = np.random.randint(0, self.policy.adv[-1].out_features, size=int(rand_mask.sum()))

        if (~rand_mask).any():
            obs_t = torch.from_numpy(obs_batch[~rand_mask]).float().to(DEVICE)
            q = self.policy(obs_t)
            actions[~rand_mask] = q.argmax(dim=1).cpu().numpy().astype(np.int64)

        return actions

    def update(self):
        if len(self.buf) < self.cfg.min_buffer:
            return

        for _ in range(int(self.cfg.grad_steps_per_update)):
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
            actions = self.act_batch(obs)
            next_obs, rewards, dones, infos = self.envs.step(actions)

            for i in range(self.envs.n_envs):
                self.buf.push(obs[i], actions[i], rewards[i], next_obs[i], float(dones[i]))
                self.episode_rewards[i] += rewards[i]

                if dones[i]:
                    self.episode_count += 1
                    self.episode_rewards[i] = 0.0

                    if (
                        self.episode_count > 0
                        and (self.episode_count % self.cfg.print_every_episodes == 0)
                        and (self.episode_count != self.last_logged_episode)
                    ):
                        print(
                            f"[STATUS] Episodes={self.episode_count}/{self.cfg.max_episodes} | "
                            f"Eps={self.eps:.3f} | Buffer={len(self.buf)}"
                        )
                        self.last_logged_episode = self.episode_count

            obs = next_obs
            self.total_transitions += self.envs.n_envs

            # treino
            if (self.total_transitions // self.envs.n_envs) % self.cfg.train_every == 0:
                self.update()

            # sync target
            if (self.total_transitions // self.envs.n_envs) % self.cfg.target_sync_every == 0:
                self.target.load_state_dict(self.policy.state_dict())

            # epsilon decay
            self.eps = max(self.cfg.eps_min, self.eps * self.cfg.eps_decay)

            # throughput
            if self.cfg.log_transitions and (self.total_transitions % (self.envs.n_envs * 500) == 0):
                dt = time.time() - t0
                sps = self.total_transitions / max(dt, 1e-6)
                print(
                    f"[train] transitions={self.total_transitions} | "
                    f"eps={self.eps:.3f} | buffer={len(self.buf)} | SPS={sps:.0f}"
                )

        elapsed = time.time() - start_time
        avg_time_per_ep = elapsed / max(1, self.episode_count)

        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = elapsed % 60

        print("\n==== TREINO FINALIZADO ====")
        print(f"Total de episódios: {self.episode_count}")
        print(f"Tempo total de treino: {h:02d}:{m:02d}:{s:05.2f}")
        print(f"Tempo médio por episódio: {avg_time_per_ep:.3f} s")
        print("====\n")

        self.envs.close()


# =====================
# COLETA DE UM EPISÓDIO (CSV) + PLOT
# =====================
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_initial_positions(env, title="Posição inicial"):
    fig, ax = plt.subplots(figsize=(8, 7))

    # limites globais para visualização (inclui área após porta)
    ax.set_xlim(env.gx_min, env.gx_max)
    ax.set_ylim(env.gy_min, env.gy_max)
    ax.set_aspect("equal")

    # sala
    sala = Rectangle(
        (env.room_xmin, env.room_ymin),
        env.room_xmax - env.room_xmin,
        env.room_ymax - env.room_ymin,
        linewidth=4,
        edgecolor="black",
        facecolor="none",
        zorder=1,
    )
    ax.add_patch(sala)

    # porta (marcação verde)
    ax.plot([env.door_wall_x, env.door_wall_x], [env.room_ymin, env.door_y_min], "k-", lw=6, zorder=2)
    ax.plot([env.door_wall_x, env.door_wall_x], [env.door_y_max, env.room_ymax], "k-", lw=6, zorder=2)
    ax.plot([env.door_wall_x, env.door_wall_x], [env.door_y_min, env.door_y_max], "g-", lw=7, zorder=3)

    # parede interna com brecha
    ax.plot([env.gap_wall_x0, env.gap_x_min], [env.gap_wall_y, env.gap_wall_y], "k-", lw=6, zorder=2)
    ax.plot([env.gap_x_max, env.gap_wall_x1], [env.gap_wall_y, env.gap_wall_y], "k-", lw=6, zorder=2)
    ax.plot([env.gap_x_min, env.gap_x_max], [env.gap_wall_y, env.gap_wall_y], "g-", lw=7, zorder=3)

    # zona de spawn (amarelo)
    spawn = Rectangle(
        (env.spawn_xmin, env.spawn_ymin),
        env.spawn_xmax - env.spawn_xmin,
        env.spawn_ymax - env.spawn_ymin,
        linewidth=2,
        edgecolor="goldenrod",
        facecolor="yellow",
        alpha=0.35,
        zorder=0,
    )
    ax.add_patch(spawn)

    # entidades
    ax.scatter(env.leader[0], env.leader[1], c="blue", s=90, label="Leader", zorder=5)
    ax.scatter(env.follower[0], env.follower[1], c="red", s=90, label="Follower", zorder=5)
    ax.scatter(env.goal[0], env.goal[1], c="green", s=140, marker="*", label="Goal", zorder=5)

    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()


def run_episode_collect(agent, episode_id: str, out_dir="logs", max_steps=400):
    os.makedirs(out_dir, exist_ok=True)

    env = agent.env if hasattr(agent, "env") else LeaderFollowerEnv()
    st = env.reset()

    plot_initial_positions(env, title=f"Cenário inicial – {episode_id.upper()}")

    filepath = os.path.join(out_dir, f"episode_{episode_id}.csv")

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "t",
                "leader_x",
                "leader_y",
                "leader_theta",
                "leader_v",
                "leader_w",
                "follower_x",
                "follower_y",
                "follower_theta",
                "follower_v",
                "follower_w",
                "reward",
                "reason",
                "passed_gap",
                "passed_door",
            ]
        )

        t = 0.0
        done = False
        steps = 0
        reason = ""

        # linha inicial
        l = env.leader
        f0 = env.follower
        writer.writerow(
            [
                t,
                l[0],
                l[1],
                l[2],
                env.v_leader,
                env.w_leader,
                f0[0],
                f0[1],
                f0[2],
                env.v_follower,
                env.w_follower,
                0.0,
                "",
                int(env.passed_gap),
                int(env.passed_door),
            ]
        )

        while not done and steps < max_steps:
            a = agent.act(st, greedy=True)
            st, r, done, info = env.step(a)

            t += env.dt * env.sub
            steps += 1
            l = env.leader
            f0 = env.follower
            reason = info.get("reason", "")

            writer.writerow(
                [
                    t,
                    l[0],
                    l[1],
                    l[2],
                    env.v_leader,
                    env.w_leader,
                    f0[0],
                    f0[1],
                    f0[2],
                    env.v_follower,
                    env.w_follower,
                    float(r),
                    reason if done else "",
                    int(env.passed_gap),
                    int(env.passed_door),
                ]
            )

    return filepath


def plot_episode(filepath, out_dir=None):
    df = pd.read_csv(filepath)
    df = df[df["leader_x"].notna()]
    t = df["t"]

    if out_dir is None:
        out_dir = os.path.dirname(filepath)
    os.makedirs(out_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(filepath))[0]

    # POSIÇÃO
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
    plt.savefig(os.path.join(out_dir, f"{base}_pose.png"), dpi=220, bbox_inches="tight")
    plt.close()

    # VELOCIDADES
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
    plt.savefig(os.path.join(out_dir, f"{base}_vel.png"), dpi=220, bbox_inches="tight")
    plt.close()

    # CHECKPOINTS (gap/door)
    plt.figure(figsize=(10, 3.6))
    plt.plot(t, df["passed_gap"], label="passed_gap")
    plt.plot(t, df["passed_door"], label="passed_door")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Flag")
    plt.title("Checkpoints (brecha/porta)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, f"{base}_checkpoints.png"), dpi=220, bbox_inches="tight")
    plt.close()

    print(f"[OK] Gráficos salvos em: {out_dir} (base: {base})")


# =====================
# Agente de avaliação (wrapper)
# =====================
class EvalAgent:
    def __init__(self, policy, env):
        self.policy = policy
        self.env = env

    def act(self, state, greedy=True):
        with torch.no_grad():
            s = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
            q = self.policy(s)
            return int(q.argmax())


def evaluate(agent, n_episodes=200, max_steps=500):
    successes = 0
    reasons = {}

    for _ in range(int(n_episodes)):
        env = agent.env
        st = env.reset()

        done = False
        steps = 0
        info = {}

        while not done and steps < max_steps:
            a = agent.act(st, greedy=True)
            st, r, done, info = env.step(a)
            steps += 1

        reason = info.get("reason", "unknown")
        reasons[reason] = reasons.get(reason, 0) + 1
        if reason == "passed_goal":
            successes += 1

    success_rate = successes / max(1, int(n_episodes))

    print("\n==== AVALIAÇÃO ====")
    print(f"Episódios: {n_episodes}")
    print(f"Sucessos: {successes}/{n_episodes}")
    print(f"Taxa de sucesso: {success_rate:.3f}")
    print("Motivos de término:")
    for k, v in reasons.items():
        print(f"  {k}: {v}")
    print("====\n")

    return success_rate, reasons


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    cfg = TrainConfig(
        gamma=0.99,
        batch=128,
        lr=1e-3,
        buffer_capacity=30000,
        min_buffer=10000,
        eps_start=1.0,
        eps_min=0.02,
        eps_decay=0.9993,
        max_episodes=500,
        train_every=1,
        target_sync_every=1500,
        grad_steps_per_update=4,
        print_every_episodes=200,
        log_transitions=False,
        hidden=512,
        depth=3,
        dropout=0.0,
    )

    print(f"Rodando com NUM_ENVS={NUM_ENVS} | DEVICE={DEVICE}")
    trainer = DQNParallelTrainer(n_envs=NUM_ENVS, cfg=cfg)

    # TREINO
    trainer.train()

    # AVALIAÇÃO + COLETA
    eval_env = LeaderFollowerEnv()
    agent = EvalAgent(trainer.policy, eval_env)

    # Coleta de um episódio demonstrativo
    csv_path = run_episode_collect(agent, episode_id="demo", out_dir="logs/demo")
    plot_episode(csv_path)

    # Avaliação estatística
    evaluate(agent, n_episodes=500)

    print("\nExecução finalizada com sucesso.")
