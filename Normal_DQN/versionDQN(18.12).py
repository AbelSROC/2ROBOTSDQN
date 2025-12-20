import os, csv
from glob import glob
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from collections import deque
import shutil



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================================
# REPLAY BUFFER
# ==========================================================
class ReplayBuffer:
    def __init__(self, cap):
        self.cap = cap
        self.buf = []
        self.pos = 0

    def push(self, s, a, r, ns, d):
        if len(self.buf) < self.cap:
            self.buf.append(None)
        self.buf[self.pos] = (s, a, r, ns, d)
        self.pos = (self.pos + 1) % self.cap

    def sample(self, batch):
        batch = random.sample(self.buf, batch)
        s, a, r, ns, d = map(np.stack, zip(*batch))
        return s, a, r, ns, d

    def __len__(self):
        return len(self.buf)


# ==========================================================
# DQN
# ==========================================================
class DQN(nn.Module):
    def __init__(self, inp, out, h=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, out)
        )

    def forward(self, x):
        return self.net(x)


# ==========================================================
# ENV: LÍDER & SEGUIDOR COM ZONA DE SEGURANÇA
# ==========================================================
# ==========================================================
# ENV: LÍDER & SEGUIDOR (goal deve ser ultrapassado)
# ==========================================================
# ==========================================================
# ENV: LÍDER & SEGUIDOR — líder deve ultrapassar o goal alinhado
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

        # Limites do gráfico
        self.xmin, self.xmax = -6, 6
        self.ymin, self.ymax = -6, 6

        # Spawn líder
        self.leader_start = np.array([self.wall_x - 4.0, 0.0, 0.0])

        # spawn seguidor
        self.spawn_radius = 0.6

        # Zona de segurança entre robôs
        self.min_safe_dist = 0.40
        self.collision_dist = 0.10

        # distância desejada
        self.desired_dist = 0.30

        self.goal = None
        self.leader = np.zeros(3)
        self.follower = np.zeros(3)
        self.buffer = deque(maxlen=self.buf_size)

        # velocidades atuais (expostas para logging)
        self.v_leader = 0.0
        self.w_leader = 0.0
        self.v_follower = 0.0
        self.w_follower = 0.0



    @staticmethod
    def wrap(a):
        return (a + np.pi) % (2*np.pi) - np.pi


    # colisão parede
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


    # dinâmica unicycle
    def _step_uni(self, pose, v, w):
        x, y, th = pose
        for _ in range(self.sub):
            x += self.dt * v * np.cos(th)
            y += self.dt * v * np.sin(th)
            th = self.wrap(th + self.dt*w)
        return np.array([x, y, th])


    # controlador SEGURO do seguidor
    def _foll_ctrl(self, pose, tgt):
        x, y, th = pose
        tx, ty = tgt

        dx = tx - x
        dy = ty - y
        dist = np.hypot(dx, dy)

        ang = np.arctan2(dy, dx)
        err = self.wrap(ang - th)

        # muito perto → para
        if dist < self.min_safe_dist:
            return 0.0, np.clip(self.kp_a * err, -1.0, 1.0)

        v = np.clip(self.kp_d * dist, 0, self.max_v_f)
        w = np.clip(self.kp_a * err, -1.5, 1.5)
        return v, w


    # estado RL
    def _state(self):
        Lx, Ly, Lth = self.leader
        Fx, Fy, Fth = self.follower
        gx, gy = self.goal

        dist_goal = np.hypot(gx - Lx, gy - Ly)
        pair = np.hypot(Lx - Fx, Ly - Fy)
        heading = np.arctan2(gy - Ly, gx - Lx)
        ang = self.wrap(Lth - heading)

        return np.array([
            gx - Lx, gy - Ly,
            dist_goal,
            pair,
            np.sin(Lth), np.cos(Lth),
            np.sin(Fth), np.cos(Fth),
            ang
        ], dtype=np.float32)


    # ==========================================================
    # RESET — goal sempre dentro do gráfico
    # ==========================================================
    def reset(self):

        self.leader = self.leader_start.copy()

        # ----- GOAL: mais distante, mas nunca fora do gráfico -----
        gx_min = self.wall_x + 2.5
        gx_max = min(self.wall_x + 4.5, self.xmax - 0.5)  # max ~5.5

        gx = np.random.uniform(gx_min, gx_max)
        gy = np.random.uniform(self.door_min + 0.1, self.door_max - 0.1)

        self.goal = np.array([gx, gy])

        # Spawn SEGURO para o seguidor
        for _ in range(40):
            ang = np.random.uniform(0, 2*np.pi)
            d = np.random.uniform(0.1, self.spawn_radius)
            x2 = self.leader[0] + d*np.cos(ang)
            y2 = self.leader[1] + d*np.sin(ang)
            if not self._wall(self.leader[:2], np.array([x2, y2])):
                self.follower = np.array([x2, y2, self.leader[2]])
                break
        else:
            self.follower = self.leader.copy()

        self.buffer.clear()
        for _ in range(self.delay+3):
            self.buffer.append(self.leader[:2].copy())

        st = self._state()
        self.last_goal = st[2]
        self.last_pair = st[3]
        self.last_ang = abs(st[8])
        return st


    # ==========================================================
    # STEP — líder deve ULTRAPASSAR o goal alinhado
    # ==========================================================
    def step(self, a):

        # -------------------------
        # ação do líder
        # -------------------------
        v = self.v_vals[a // 3]
        w = self.w_vals[a % 3]

        oldL = self.leader.copy()
        oldF = self.follower.copy()

        # mover líder
        self.leader = self._step_uni(self.leader, v, w)
        self.buffer.append(self.leader[:2])

        # -------------------------
        # controlador do seguidor
        # -------------------------
        tgt = self.buffer[-1 - self.delay]
        v2, w2 = self._foll_ctrl(self.follower, tgt)

        # mover seguidor
        self.follower = self._step_uni(self.follower, v2, w2)

        # -------------------------
        # LOG DAS VELOCIDADES (AGORA CORRETO)
        # -------------------------
        self.v_leader = float(v)
        self.w_leader = float(w)
        self.v_follower = float(v2)
        self.w_follower = float(w2)
        # --------------------------------------------------
        # Verificações de mapa
        # --------------------------------------------------
        for x, y in [(self.leader[0], self.leader[1]),
                     (self.follower[0], self.follower[1])]:
            if x < self.xmin or x > self.xmax or y < self.ymin or y > self.ymax:
                return self._state(), -200, True, {"reason": "out"}

        if self._wall(oldL[:2], self.leader[:2]) or self._wall(oldF[:2], self.follower[:2]):
            return self._state(), -250, True, {"reason": "wall"}

        # --------------------------------------------------
        # Recompensa
        # --------------------------------------------------
        st = self._state()
        dist_goal = st[2]
        pair = st[3]
        ang = abs(st[8])

        reward = -0.05

        # aproximar do goal
        reward += 8 * (self.last_goal - dist_goal)

        # manter distância correta
        reward += 2 * (abs(self.last_pair - self.desired_dist)
                       - abs(pair - self.desired_dist))

        # penaliza desalinhamento angular
        reward -= 0.3 * ang

        # penalização lateral (NOVA)
        lat_error = abs(self.leader[1] - self.goal[1])
        reward -= 2.0 * lat_error

        # punir sair do corredor
        if self.leader[1] < self.door_min - 0.3 or self.leader[1] > self.door_max + 0.3:
            reward -= 60

        # muito perto
        if pair < self.min_safe_dist:
            reward -= 40

        # colisão
        if pair < self.collision_dist:
            return st, -150, True, {"reason": "robot_collision"}

        # bônus porta
        if oldL[0] < self.wall_x < self.leader[0]:
            if self.door_min <= self.leader[1] <= self.door_max:
                reward += 80

        # ==========================================================
        # NADA de recompensa por "chegar perto"
        # ==========================================================

        # ==========================================================
        # TERMINA APENAS SE O LÍDER ULTRAPASSAR O GOAL ALINHADO
        # ==========================================================
        if self.leader[0] > self.goal[0] + 0.1:  # passou 5 cm
            reward += 180  # incentivo forte

            # alinhamento em Y
            if abs(self.leader[1] - self.goal[1]) < 0.2:
                reward += 80
            else:
                reward -= 80

            # formação: seguidor → goal → líder
            if self.follower[0] < self.goal[0] < self.leader[0]:
                reward += 120
            else:
                reward -= 80

            return st, reward, True, {"reason": "passed_goal"}

        # atualizar históricos
        self.last_goal = dist_goal
        self.last_pair = pair
        self.last_ang = ang

        return st, reward, False, {}


    # posições para vídeo
    def poses(self):
        return self.leader.copy(), self.follower.copy(), self.goal.copy()

# ==========================================================
# AGENTE DQN
# ==========================================================
class Agent:
    def __init__(self):
        self.env = LeaderFollowerEnv()

        s = self.env.reset()
        self.sd = len(s)
        self.ad = self.env.na

        self.gamma = 0.99
        self.batch = 64
        self.lr = 1e-3

        self.buf = ReplayBuffer(100000)
        self.min_buf = 2000

        self.policy = DQN(self.sd, self.ad).to(DEVICE)
        self.target = DQN(self.sd, self.ad).to(DEVICE)
        self.target.load_state_dict(self.policy.state_dict())

        self.opt = optim.Adam(self.policy.parameters(), lr=self.lr)

        self.eps = 1.0
        self.eps_min = 0.05
        self.eps_decay = 0.9993

        self.global_step = 0
        self.target_int = 1500

        self.max_steps = 300

    def act(self, st, greedy=False):
        if not greedy and random.random() < self.eps:
            return random.randrange(self.ad)
        st = torch.FloatTensor(st).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            q = self.policy(st)
        return int(q.argmax())

    def update(self):
        if len(self.buf) < self.min_buf:
            return

        s, a, r, ns, d = self.buf.sample(self.batch)
        s = torch.FloatTensor(s).to(DEVICE)
        ns = torch.FloatTensor(ns).to(DEVICE)
        a = torch.LongTensor(a).to(DEVICE).view(-1, 1)
        r = torch.FloatTensor(r).to(DEVICE).view(-1, 1)
        d = torch.FloatTensor(d).to(DEVICE).view(-1, 1)

        q = self.policy(s).gather(1, a)
        max_next = self.target(ns).max(1)[0].unsqueeze(1).detach()
        target = r + (1-d)*self.gamma*max_next

        loss = nn.MSELoss()(q, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def train(self, ep=300):
        for e in range(1, ep+1):
            st = self.env.reset()
            tot = 0

            for _ in range(self.max_steps):
                a = self.act(st)
                ns, r, done, _ = self.env.step(a)
                self.buf.push(st, a, r, ns, float(done))
                st = ns
                tot += r

                self.update()

                self.global_step += 1
                if self.global_step % self.target_int == 0:
                    self.target.load_state_dict(self.policy.state_dict())

                if done:
                    break

            self.eps = max(self.eps_min, self.eps * self.eps_decay)

            if e % 200 == 0:
                print(f"EP {e} | Reward={tot:.1f} | eps={self.eps:.3f}")

    def simulate(self, steps=300):
        st = self.env.reset()

        trajL, trajF = [], []

        L, F, _ = self.env.poses()
        trajL.append(L.copy())
        trajF.append(F.copy())

        for _ in range(steps):
            a = self.act(st, greedy=True)
            ns, r, done, _ = self.env.step(a)
            st = ns

            L, F, _ = self.env.poses()
            trajL.append(L.copy())
            trajF.append(F.copy())

            if done:
                break

        if len(trajL) == 0:
            L, F, _ = self.env.poses()
            trajL, trajF = [L], [F]

        return np.array(trajL), np.array(trajF), self.env.goal, self.env


# ==========================================================
# GERAR VÍDEO (MP4 ou GIF)
# ==========================================================
def gerar_video(L, F, goal, env, nome="simulacao.mp4", freeze_seconds=3):

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_xlim(env.xmin, env.xmax)
    ax.set_ylim(env.ymin, env.ymax)
    ax.set_aspect("equal")

    # parede
    xw = env.wall_x
    ax.plot([xw, xw], [env.wall_min, env.door_min], "k-", lw=6)
    ax.plot([xw, xw], [env.door_max, env.wall_max], "k-", lw=6)
    ax.plot([xw, xw], [env.door_min, env.door_max], "g-", lw=8)

    ax.scatter(goal[0], goal[1], s=200, marker="*", c="green")

    lead_pt, = ax.plot([], [], "bo", markersize=8)
    foll_pt, = ax.plot([], [], "ro", markersize=8)
    lead_tr, = ax.plot([], [], "b-")
    foll_tr, = ax.plot([], [], "r-")

    def update(i):
        lead_pt.set_data([L[i, 0]], [L[i, 1]])
        foll_pt.set_data([F[i, 0]], [F[i, 1]])
        lead_tr.set_data(L[:i+1, 0], L[:i+1, 1])
        foll_tr.set_data(F[:i+1, 0], F[:i+1, 1])
        return lead_pt, foll_pt, lead_tr, foll_tr

    print("Gerando vídeo...")

    has_ffmpeg = shutil.which("ffmpeg") is not None
    n_frames = len(L)

    if has_ffmpeg:
        fps = 30
        extra_frames = int(freeze_seconds * fps)
        writer = animation.FFMpegWriter(fps=fps)
        with writer.saving(fig, nome, 120):
            for i in range(n_frames + extra_frames):
                idx = min(i, n_frames - 1)  # depois que acabar, fica no último frame
                update(idx)
                writer.grab_frame()
        print(f"Vídeo salvo: {nome}")
    else:
        print("⚠ FFmpeg não encontrado! Salvando como GIF...")
        fps = 20
        extra_frames = int(freeze_seconds * fps)
        gif_name = nome.replace(".mp4", ".gif")
        writer = animation.PillowWriter(fps=fps)
        with writer.saving(fig, gif_name, 100):
            for i in range(n_frames + extra_frames):
                idx = min(i, n_frames - 1)
                update(idx)
                writer.grab_frame()
        print(f"GIF salvo: {gif_name}")


# ==========================================================
# MAIN
# ==========================================================
def set_start_easy(env):
    env.leader = env.leader_start.copy()
    env.follower = env.leader.copy()
    env.follower[0] -= 0.4
    env.follower[2] = env.leader[2]

def set_start_medium(env, angle_deg=45, distance=1.2):
    import numpy as np
    theta = np.deg2rad(angle_deg)
    env.leader = env.leader_start.copy()
    env.leader[2] = theta
    env.leader[0] -= distance * np.cos(theta)
    env.leader[1] -= distance * np.sin(theta)
    env.follower = env.leader.copy()
    env.follower[0] -= 0.6*np.cos(theta)
    env.follower[1] -= 0.6*np.sin(theta)

def set_start_hard(env, angle_deg=90, distance=1.2):
    import numpy as np
    theta = np.deg2rad(angle_deg)
    env.leader = env.leader_start.copy()
    env.leader[2] = theta
    env.leader[0] -= distance * np.cos(theta)
    env.leader[1] -= distance * np.sin(theta)
    env.follower = env.leader.copy()
    env.follower[0] -= 0.6*np.cos(theta)
    env.follower[1] -= 0.6*np.sin(theta)
def evaluate_agent(agent, set_start_fn, n=10):
    import numpy as np
    successes = 0
    for i in range(n):
        st = agent.env.reset()
        set_start_fn(agent.env)  # força o cenário aqui
        done = False
        steps = 0
        while not done:
            a = agent.act(st, greedy=True)
            st, r, done, info = agent.env.step(a)
            steps += 1
        print(f"Episódio {i}: reason={info['reason']} steps={steps}")
        if info["reason"] == "passed_goal":
            successes += 1
    print("Taxa de sucesso:", successes / n)

def evaluate_difficulty(agent, set_start_fn, name, n_episodes=10):
    successes = 0
    reasons = {}

    for i in range(n_episodes):
        st = agent.env.reset()
        set_start_fn(agent.env)
        st = agent.env._state()

        done = False
        steps = 0
        info = {}

        while not done and steps < agent.max_steps:
            a = agent.act(st, greedy=True)
            st, r, done, info = agent.env.step(a)
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
# ==========================================================
# GERAR VÍDEO (MP4 ou GIF)
# ==========================================================
def gerar_video(L, F, goal, env, nome="simulacao.mp4", freeze_seconds=3):

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_xlim(env.xmin, env.xmax)
    ax.set_ylim(env.ymin, env.ymax)
    ax.set_aspect("equal")

    # parede
    xw = env.wall_x
    ax.plot([xw, xw], [env.wall_min, env.door_min], "k-", lw=6)
    ax.plot([xw, xw], [env.door_max, env.wall_max], "k-", lw=6)
    ax.plot([xw, xw], [env.door_min, env.door_max], "g-", lw=8)

    ax.scatter(goal[0], goal[1], s=200, marker="*", c="green")

    lead_pt, = ax.plot([], [], "bo", markersize=8)
    foll_pt, = ax.plot([], [], "ro", markersize=8)
    lead_tr, = ax.plot([], [], "b-")
    foll_tr, = ax.plot([], [], "r-")

    def update(i):
        lead_pt.set_data([L[i, 0]], [L[i, 1]])
        foll_pt.set_data([F[i, 0]], [F[i, 1]])
        lead_tr.set_data(L[:i+1, 0], L[:i+1, 1])
        foll_tr.set_data(F[:i+1, 0], F[:i+1, 1])
        return lead_pt, foll_pt, lead_tr, foll_tr

    print("Gerando vídeo...")

    has_ffmpeg = shutil.which("ffmpeg") is not None
    n_frames = len(L)

    if has_ffmpeg:
        fps = 30
        extra_frames = int(freeze_seconds * fps)
        writer = animation.FFMpegWriter(fps=fps)
        with writer.saving(fig, nome, 120):
            for i in range(n_frames + extra_frames):
                idx = min(i, n_frames - 1)  # depois que acabar, fica no último frame
                update(idx)
                writer.grab_frame()
        print(f"Vídeo salvo: {nome}")
    else:
        print("⚠ FFmpeg não encontrado! Salvando como GIF...")
        fps = 20
        extra_frames = int(freeze_seconds * fps)
        gif_name = nome.replace(".mp4", ".gif")
        writer = animation.PillowWriter(fps=fps)
        with writer.saving(fig, gif_name, 100):
            for i in range(n_frames + extra_frames):
                idx = min(i, n_frames - 1)
                update(idx)
                writer.grab_frame()
        print(f"GIF salvo: {gif_name}")

def run_episode_collect(agent, set_start_fn, episode_id, out_dir="logs", max_steps=300):
    os.makedirs(out_dir, exist_ok=True)
    env = agent.env
    st = env.reset()
    set_start_fn(env)  # força cenário easy/medium/hard
    st = env._state()
    plot_initial_positions(env, title=f"Cenário inicial – {set_start_fn.__name__}")



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
        reason = ""
        steps = 0

        # registra estado inicial (t=0)
        l = env.leader; f = env.follower
        writer.writerow([
            t,
            l[0], l[1], l[2],
            env.v_leader, env.w_leader,
            f[0], f[1], f[2],
            env.v_follower, env.w_follower,
            0.0, ""
        ])

        while not done and steps < max_steps:
            a = agent.act(st, greedy=True)
            ns, r, done, info = env.step(a)
            st = ns
            t += env.dt * env.sub
            steps += 1
            l = env.leader; f = env.follower
            reason = info.get("reason", "")
            writer.writerow([
                t,
                l[0], l[1], l[2],
                env.v_leader, env.w_leader,
                f[0], f[1], f[2],
                env.v_follower, env.w_follower,
                r, reason if done else ""
            ])

    return filepath

def plot_episode(filepath, out_dir=None):
    import pandas as pd
    import os
    import matplotlib.pyplot as plt

    # =========================
    # Carregar CSV
    # =========================
    df = pd.read_csv(filepath)
    df = df[df["leader_x"].notna()]

    t = df["t"]

    if out_dir is None:
        out_dir = os.path.dirname(filepath)
    os.makedirs(out_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(filepath))[0]

    # =========================
    # 1) x vs tempo
    # =========================
    plt.figure(figsize=(8,5))
    plt.plot(t, df["leader_x"], label="Robô B (Líder)")
    plt.plot(t, df["follower_x"], label="Robô A (Seguidor)")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Posição x (m)")
    plt.title("Posição x vs Tempo")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{base}_x_vs_t.png"), dpi=300)
    plt.close()

    # =========================
    # 2) y vs tempo
    # =========================
    plt.figure(figsize=(8,5))
    plt.plot(t, df["leader_y"], label="Robô B (Líder)")
    plt.plot(t, df["follower_y"], label="Robô A (Seguidor)")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Posição y (m)")
    plt.title("Posição y vs Tempo")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{base}_y_vs_t.png"), dpi=300)
    plt.close()

    # =========================
    # 3) theta vs tempo
    # =========================
    plt.figure(figsize=(8,5))
    plt.plot(t, df["leader_theta"], label="Robô B (Líder)")
    plt.plot(t, df["follower_theta"], label="Robô A (Seguidor)")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Orientação θ (rad)")
    plt.title("Orientação θ vs Tempo")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{base}_theta_vs_t.png"), dpi=300)
    plt.close()

    # =========================
    # 4) velocidade linear vs tempo
    # =========================
    plt.figure(figsize=(8,5))
    plt.plot(t, df["leader_v"], label="Robô B (Líder)")
    plt.plot(t, df["follower_v"], label="Robô A (Seguidor)")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Velocidade linear v (m/s)")
    plt.title("Velocidade Linear vs Tempo")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{base}_v_vs_t.png"), dpi=300)
    plt.close()

    # =========================
    # 5) velocidade angular vs tempo
    # =========================
    plt.figure(figsize=(8,5))
    plt.plot(t, df["leader_w"], label="Robô B (Líder)")
    plt.plot(t, df["follower_w"], label="Robô A (Seguidor)")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Velocidade angular ω (rad/s)")
    plt.title("Velocidade Angular vs Tempo")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{base}_w_vs_t.png"), dpi=300)
    plt.close()

    print(f"[OK] Gráficos salvos em: {out_dir}")




def plot_initial_positions(env, title="Posição inicial"):
    plt.figure(figsize=(6,6))
    plt.xlim(env.xmin, env.xmax)
    plt.ylim(env.ymin, env.ymax)
    plt.gca().set_aspect("equal")

    # parede
    xw = env.wall_x
    plt.plot([xw, xw], [env.wall_min, env.door_min], "k-", lw=4)
    plt.plot([xw, xw], [env.door_max, env.wall_max], "k-", lw=4)
    plt.plot([xw, xw], [env.door_min, env.door_max], "g-", lw=6)

    # robôs
    plt.scatter(env.leader[0], env.leader[1], c="blue", s=80, label="Leader")
    plt.scatter(env.follower[0], env.follower[1], c="red", s=80, label="Follower")

    # goal
    plt.scatter(env.goal[0], env.goal[1], c="green", s=120, marker="*", label="Goal")

    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":

    # =========================
    # ETAPA 1 — CONTROLE DE EP
    # =========================
 
    agent = Agent()
    agent.train(ep=1000) # <- altera os eps de treino IMPORTANTE

    # =========================
    # ETAPA 2 — COLETA + GRÁFICOS
    # =========================
    scenarios = [
        ("easy", set_start_easy),
        ("medium", set_start_medium),
        ("hard", set_start_hard),
    ]

    for name, start_fn in scenarios:
        print(f"\n=== ETAPA 2 | {name.upper()} ===")

        csv_path = run_episode_collect(
            agent,
            start_fn,
            episode_id=name,
            out_dir=f"logs/{name}"
        )

        plot_episode(csv_path)  # <- sem isso não tem gráfico
    for name, fn in scenarios:
        evaluate_difficulty(agent, fn, name, n_episodes=500)   # <- altera os eps de treino IMPORTANTE

    # =========================
    # ETAPA 3 — VÍDEOS / GIFS POR DIFICULDADE
    # =========================
    print("\nGerando vídeos por dificuldade...\n")

    scenarios = [
        ("easy", set_start_easy),
        ("medium", set_start_medium),
        ("hard", set_start_hard),
    ]

    for name, start_fn in scenarios:
        print(f" Gerando vídeo: {name.upper()}")

        # força cenário
        agent.env.reset()
        start_fn(agent.env)

        # simula com política treinada
        L, F, g, env = agent.simulate(steps=300)

        # gera vídeo / gif
        gerar_video(
            L, F, g, env,
            nome=f"sim_{name}.mp4",   # vira .gif se não tiver ffmpeg
            freeze_seconds=3
        )

    print("\n Todos os vídeos foram gerados.")
