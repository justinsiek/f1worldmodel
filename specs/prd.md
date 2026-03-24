2D F1 World Model Spec
Goal

Build a top-down 2D F1-style racing simulator where a car learns an action-conditioned latent world model and uses it for MPC planning.

Core loop
Simulate car on track
Render local raster observation around ego car
Encode observation into latent
Predict future latents conditioned on candidate actions
Score predicted futures
Execute first action from best sequence
Train on collected trajectories
Environment
Car state
state = {
    "x": float,
    "y": float,
    "theta": float,
    "velocity": float,
}
Action

Continuous:

action = {
    "steer": float,     # [-1, 1]
    "throttle": float,  # [0, 1]
    "brake": float,     # [0, 1]
}
Dynamics

Use a simple kinematic or semi-dynamic car model:

steer updates heading
throttle increases velocity
brake decreases velocity
heading + velocity update position
optional grip/slip scaling at high speed
Track

Track contains:

drivable area mask
wall / boundary mask
centerline or waypoint sequence
lap progress definition
Episode ends
lap completed
off track too long
max steps reached
Observation
Format
obs = {
    "raster": float32[C, H, W],
    "aux": float32[A],
}
V1 raster

Use ego-centered, ego-rotated local crop:

centered on car
rotated so car always faces up

Suggested shape:

raster.shape = (3, 64, 64)
Raster channels
channel 0: drivable area
channel 1: walls / boundaries
channel 2: ego car mask
Aux features
aux = [speed]

Optional later:

aux = [speed, steering_angle]
Reward
Per-step
reward =
    + 0.02 * progress_delta
    - 0.5  * off_track_event
    - 0.005
Terminal
+1.0 for lap completion

Optional later:

instability penalty
control smoothness penalty
Dataset
Collect trajectories from 19 geometrically distinct physics profiles to mathematically guarantee robust latent bounds, including:
- Expert baselines (`scripted`, `noisy`)
- Extreme kinematics (`kamikaze`, `brakepump`, `donut`)
- Lateral aerodynamics (`wobble`, `drift`, `panic`)
- Symmetrical bias correction (`rightbias`, `leftbias`, `sinewave`)
Store
transition = {
    "obs": obs_t,
    "action": action_t,
    "next_obs": obs_t1,
    "reward": float,
    "done": bool,
}
Model
Chosen model

An Action-Conditional **JEPA** (Joint Embedding Predictive Architecture). The model maps 64x64 sensory pixels into a pure Latent MDP, entirely bypassing physical Autoencoder pixel-reconstruction while organically learning 100% of the aerodynamic and kinematic matrix.

Encoder

Inputs:

raster
aux

Architecture:

CNN for raster
MLP for aux
concatenate
linear projection to latent

Output:

z_t = encoder(raster_t, aux_t)

Suggested latent size:

latent_dim = 64
Target encoder
EMA copy of encoder
no gradients
used to create target latent
z_target = target_encoder(raster_t1, aux_t1)
Action encoder

Small MLP:

a_emb = action_encoder(action_t)
Predictor

Input:

current latent
action embedding

Output:

next latent
z_pred = predictor(z_t, a_emb)

Use:

MLP for simplest version
GRU later if needed
Optional heads

Train small heads on latent for planner scoring:

progress head
offtrack head
heading error head
speed head
Training
Main loss
loss = mse(z_pred, z_target)
Multi-step training

Unroll predictor for horizon H:

z = z_t
loss = 0
for k in range(H):
    z = predictor(z, action_k)
    loss += mse(z, z_target_k)
Training order
1-step prediction
short multi-step rollout
longer rollout after stable training
Planner
Method

Cross-Entropy Method (CEM) Random-Shooting Latent MPC.
The planner operates **100% implicitly without heuristic guardrails** (No AEB intercepts, no Lane-Keeping PID loops, no Action Seeding). The robustness of the 19-policy dataset mathematically guarantees the predictive viability of raw sequential action extraction.

At each step
encode current observation to z_t
sample N action sequences of length H
rollout predictor for each sequence
score each imagined trajectory
choose best sequence
execute first action only
repeat next timestep
Sampled action tensor
actions.shape = (N, H, 3)
Planner scoring

Use latent heads:

score = sum(progress_pred) - lambda_offtrack * sum(offtrack_pred)

Optional later:

heading alignment
instability penalty
speed target bonus
Versions
V1
single track
solo car
no opponents
V2
multiple tracks
different corner geometries
V3
tighter turns
higher speed sensitivity
grip matters more
V4
varying grip / track conditions
V5
add one opponent car
add raster channel for opponents
V6
multiple opponents
Hyperparameters
Model
latent_dim
encoder width/depth
predictor size
rollout horizon
Planner
N = number of candidate sequences
H = planning horizon
score weights
Data
random vs expert ratio
track diversity
Environment
max speed
grip
track width
Baselines
scripted controller
behavior cloning policy
oracle planner using true state
Metrics
lap completion rate
average lap progress
off-track rate
average reward
latent prediction error vs horizon
Recommended V1 defaults
raster_shape = (3, 64, 64)
aux_dim = 16              # speed + 15 Lidar rays
latent_dim = 64
planner_candidates = 400
planner_horizon = 25
model = "Action-Conditional JEPA (CNN + MLP)"
control = "continuous"
Required modules
env/ car dynamics, track, raster renderer
data/ trajectory collection and dataset
models/ encoder, target encoder, action encoder, predictor, heads
planner/ random-shooting MPC
train/ training loop
eval/ metrics and rollouts
configs/ hyperparameters