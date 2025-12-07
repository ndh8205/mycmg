function results = run_mppi_att_sim(mppi_params, params, t_end, verbose)
% run_mppi_att_sim: MPPI attitude control simulation (for tuning)
%
% Inputs:
%   mppi_params - MPPI parameter struct
%   params      - drone parameters
%   t_end       - simulation duration [s]
%   verbose     - print output (default: false)
%
% Outputs:
%   results     - struct with performance metrics

if nargin < 4
    verbose = false;
end

%% Disturbance settings
dist_preset = 'nominal';
[params, dist_state] = dist_init(params, dist_preset);
params_true = params;

%% Simulation settings
sim_hz = 1000;
dt_sim = 1/sim_hz;
t = 0:dt_sim:t_end;
N_sim = length(t);

ctrl_hz = 50;
ctrl_decimation = sim_hz / ctrl_hz;
dt_ctrl = 1 / ctrl_hz;

omega_bar2RPM = 60 / (2 * pi);

%% Process noise covariance
Q = zeros(9,9);

%% Drone parameters for MEX
drone_params.m = single(params.drone.body.m);
drone_params.Jxx = single(params.drone.body.J(1,1));
drone_params.Jyy = single(params.drone.body.J(2,2));
drone_params.Jzz = single(params.drone.body.J(3,3));
drone_params.g = single(params.env.g);
drone_params.k_T = single(params.drone.motor.k_T);
drone_params.k_M = single(params.drone.motor.k_M);
drone_params.L = single(params.drone.body.L);
drone_params.omega_max = single(params.drone.motor.omega_b_max);
drone_params.omega_min = single(params.drone.motor.omega_b_min);

%% Initial state
m = params.drone.body.m;
g = params.env.g;
k_T = params.drone.motor.k_T;
n_motor = params.drone.body.n_motor;

omega_hover = sqrt(m * g / (n_motor * k_T));

% Initial euler: roll=20, pitch=15, yaw=10 deg
euler0 = deg2rad([20; 15; 10]);
q0 = GetQUAT(euler0(3), euler0(2), euler0(1));
q0 = q0(:);

x0 = zeros(28, 1);
x0(1:3)   = [0; 0; -10];
x0(4:6)   = [0; 0; 0];
x0(7:10)  = q0;
x0(11:13) = [0; 0; 0];
x0(14:19) = omega_hover * ones(6,1);
x0(20:28) = zeros(9, 1);

%% Desired states
pos_des = [0; 0; -10];
yaw_des = 0;

%% Initialize MPPI state
mppi_state.u_seq = single(omega_hover * ones(6, mppi_params.N));
mppi_state.pos_des = single(pos_des);
mppi_state.yaw_des = single(yaw_des);

%% Data storage
X = zeros(28, N_sim);
U = zeros(6, N_sim);
X(:,1) = x0;
u_current = omega_hover * ones(6,1);

% Diagnostics
ess_log = [];
sat_log = [];
exec_time_log = [];

%% Simulation loop
diverged = false;
diverge_time = t_end;

for k = 1:N_sim-1
    x_k = X(:,k);
    
    % Check for divergence (only extreme cases)
    euler_k = Quat2Euler(x_k(7:10));
    if any(isnan(x_k)) || any(isinf(x_k)) || ...
       max(abs(rad2deg(euler_k(1:2)))) > 89 || ...  % roll/pitch > 89 deg (flip)
       -x_k(3) < -50 || -x_k(3) > 500 || ...         % altitude way out
       norm(x_k(1:2)) > 200                           % XY drift > 200m
        diverged = true;
        diverge_time = t(k);
        % Fill remaining with last valid
        X(:, k+1:end) = repmat(x_k, 1, N_sim-k);
        break;
    end
    
    % MPPI control update
    if mod(k-1, ctrl_decimation) == 0
        x_mppi = single([x_k(1:3); x_k(4:6); x_k(7:10); x_k(11:13)]);
        mppi_state.u_seq(:, 1:end-1) = mppi_state.u_seq(:, 2:end);
        
        tic_mppi = tic;
        [u_opt, u_seq_new, stats] = hexa_mppi_mex(...
            x_mppi, mppi_state.u_seq, mppi_state.pos_des, mppi_state.yaw_des, ...
            drone_params, mppi_params);
        exec_time = toc(tic_mppi);
        
        u_current = double(u_opt);
        mppi_state.u_seq = u_seq_new;
        
        ess_log(end+1) = stats.effective_sample_size;
        sat_log(end+1) = stats.saturation_ratio;
        exec_time_log(end+1) = exec_time;
    end
    
    % Apply control
    u = u_current;
    omega_max = params.drone.motor.omega_b_max;
    omega_min = params.drone.motor.omega_b_min;
    u = max(min(u, omega_max), omega_min);
    U(:,k) = u;
    
    % Integration
    [x_next, ~, ~, ~, ~, ~, dist_state] = srk4(@drone_dynamics, x_k, u, Q, dt_sim, params_true, k, dt_sim, t(k), dist_state);
    
    x_next(7:10) = x_next(7:10) / norm(x_next(7:10));
    x_next(7:10) = EnsQuatCont(x_next(7:10), x_k(7:10));
    
    X(:,k+1) = x_next;
end
U(:,end) = U(:,end-1);

%% Extract Euler angles
euler = zeros(3, N_sim);
for k = 1:N_sim
    euler(:,k) = Quat2Euler(X(7:10,k));
end

%% Compute metrics

% Settling time (2 deg threshold)
settling_threshold = deg2rad(2);
settling_time = zeros(3,1);
for i = 1:3
    settled_idx = find(abs(euler(i,:)) < settling_threshold, 1, 'first');
    if ~isempty(settled_idx)
        if all(abs(euler(i, settled_idx:end)) < settling_threshold)
            settling_time(i) = t(settled_idx);
        else
            settling_time(i) = t_end;
        end
    else
        settling_time(i) = t_end;
    end
end

% Steady-state evaluation (last 50% of simulation)
ss_start_idx = round(N_sim * 0.5);
euler_ss = euler(:, ss_start_idx:end);
alt_ss = -X(3, ss_start_idx:end);
pos_xy_ss = X(1:2, ss_start_idx:end);

% RMSE attitude (steady-state) [deg]
rmse_roll = sqrt(mean(rad2deg(euler_ss(1,:)).^2));
rmse_pitch = sqrt(mean(rad2deg(euler_ss(2,:)).^2));
rmse_yaw = sqrt(mean(rad2deg(euler_ss(3,:)).^2));
rmse_att = sqrt(mean(rad2deg(euler_ss(:)).^2));

% RMSE altitude (steady-state) [m]
alt_des_val = -pos_des(3);
rmse_alt = sqrt(mean((alt_ss - alt_des_val).^2));

% Max attitude error (steady-state) [deg]
max_att_ss = max(abs(rad2deg(euler_ss(:))));

% XY drift (max during steady-state) [m]
xy_drift_ss = max(vecnorm(pos_xy_ss));
xy_drift_final = norm(X(1:2,end));

% Oscillation metric (attitude rate std in steady-state)
omega_ss = X(11:13, ss_start_idx:end);
oscillation = mean(std(rad2deg(omega_ss), 0, 2));

%% Phase Portrait Envelope Metrics
% Envelope thresholds (relaxed for MPPI)
env.roll_ang = 2.0;    % deg (was 0.1)
env.roll_rate = 5.0;   % deg/s (was 0.5)
env.pitch_ang = 2.0;   % deg
env.pitch_rate = 5.0;  % deg/s
env.yaw_ang = 5.0;     % deg (was 0.2)
env.yaw_rate = 5.0;    % deg/s
env.alt_err = 0.5;     % m (was 0.05)
env.alt_vel = 1.0;     % m/s (was 0.1)

% Compute NED velocity for altitude rate
vel_ned_all = zeros(3, N_sim);
for kk = 1:N_sim
    R_b2n = GetDCM_QUAT(X(7:10,kk));
    vel_ned_all(:,kk) = R_b2n * X(4:6,kk);
end
alt_vel_all = -vel_ned_all(3,:);  % positive up

% Normalized distance to target (d < 1 means inside envelope)
d_roll = sqrt((rad2deg(euler(1,:))/env.roll_ang).^2 + (rad2deg(X(11,:))/env.roll_rate).^2);
d_pitch = sqrt((rad2deg(euler(2,:))/env.pitch_ang).^2 + (rad2deg(X(12,:))/env.pitch_rate).^2);
d_yaw = sqrt((rad2deg(euler(3,:))/env.yaw_ang).^2 + (rad2deg(X(13,:))/env.yaw_rate).^2);
alt_err_all = -X(3,:) - alt_des_val;
d_alt = sqrt((alt_err_all/env.alt_err).^2 + (alt_vel_all/env.alt_vel).^2);

% Envelope metrics per channel
[env_roll] = compute_envelope_metrics(d_roll, t);
[env_pitch] = compute_envelope_metrics(d_pitch, t);
[env_yaw] = compute_envelope_metrics(d_yaw, t);
[env_alt] = compute_envelope_metrics(d_alt, t);

% Max overshoot (entire simulation)
max_overshoot = max(abs(rad2deg(euler(:))));

% ESS statistics
ess_mean = mean(ess_log);
ess_min = min(ess_log);

% Saturation statistics
sat_mean = mean(sat_log);
sat_max = max(sat_log);

% Execution time
exec_mean = mean(exec_time_log);

%% Pack results
results.diverged = diverged;
results.diverge_time = diverge_time;

results.settling_time = settling_time;
results.settling_time_avg = mean(settling_time);
results.max_overshoot = max_overshoot;

% Steady-state metrics
results.rmse_roll = rmse_roll;
results.rmse_pitch = rmse_pitch;
results.rmse_yaw = rmse_yaw;
results.rmse_att = rmse_att;
results.rmse_alt = rmse_alt;
results.max_att_ss = max_att_ss;
results.xy_drift_ss = xy_drift_ss;
results.xy_drift_final = xy_drift_final;
results.oscillation = oscillation;

% Legacy
results.xy_drift = xy_drift_final;
results.alt_error = abs(-X(3,end) - alt_des_val);
results.final_att_error = norm(rad2deg(euler(:,end)));

% Envelope metrics
results.env_roll = env_roll;
results.env_pitch = env_pitch;
results.env_yaw = env_yaw;
results.env_alt = env_alt;

results.ess_mean = ess_mean;
results.ess_min = ess_min;
results.sat_mean = sat_mean;
results.sat_max = sat_max;
results.exec_mean = exec_mean;

% Raw data
results.t = t;
results.euler = euler;
results.X = X;

if verbose
    if diverged
        fprintf('*** DIVERGED at t=%.2fs ***\n', diverge_time);
    end
    fprintf('Settling: [%.2f, %.2f, %.2f] s, Avg: %.2f s\n', ...
        settling_time(1), settling_time(2), settling_time(3), mean(settling_time));
    fprintf('RMSE(ss): att=%.3f deg, alt=%.4f m\n', rmse_att, rmse_alt);
    fprintf('Max(ss): att=%.2f deg, drift=%.3f m\n', max_att_ss, xy_drift_ss);
    fprintf('Oscillation: %.2f deg/s, Overshoot: %.2f deg\n', oscillation, max_overshoot);
    fprintf('ESS: %.1f (min %.1f), Sat: %.1f%% (max %.1f%%)\n', ...
        ess_mean, ess_min, sat_mean*100, sat_max*100);
    fprintf('--- Envelope (d<1) ---\n');
    fprintf('Roll:  t_enter=%.2fs, stay=%.1f%%, final_d=%.2f\n', ...
        env_roll.t_enter, env_roll.stay_ratio*100, env_roll.final_d);
    fprintf('Pitch: t_enter=%.2fs, stay=%.1f%%, final_d=%.2f\n', ...
        env_pitch.t_enter, env_pitch.stay_ratio*100, env_pitch.final_d);
    fprintf('Yaw:   t_enter=%.2fs, stay=%.1f%%, final_d=%.2f\n', ...
        env_yaw.t_enter, env_yaw.stay_ratio*100, env_yaw.final_d);
    fprintf('Alt:   t_enter=%.2fs, stay=%.1f%%, final_d=%.2f\n', ...
        env_alt.t_enter, env_alt.stay_ratio*100, env_alt.final_d);
end

end

%% ========== HELPER FUNCTION ==========
function env = compute_envelope_metrics(d, t)
% compute_envelope_metrics: Phase portrait envelope analysis
%
% d < 1 means inside envelope
% 
% Outputs:
%   env.t_enter     - first time d < 1 [s]
%   env.stay_ratio  - ratio of time d < 1 after first entry
%   env.final_d     - final normalized distance
%   env.max_d_after - max d after first entry (chattering indicator)
%   env.mean_d_ss   - mean d in last 50%

    t_end = t(end);
    N = length(d);
    
    % Find first entry (d < 1)
    inside = d < 1;
    first_idx = find(inside, 1, 'first');
    
    if isempty(first_idx)
        % Never entered
        env.t_enter = t_end;
        env.stay_ratio = 0;
        env.final_d = d(end);
        env.max_d_after = max(d);
        env.mean_d_ss = mean(d(round(N*0.5):end));
    else
        env.t_enter = t(first_idx);
        
        % Stay ratio after first entry
        after_entry = inside(first_idx:end);
        env.stay_ratio = sum(after_entry) / length(after_entry);
        
        % Final distance
        env.final_d = d(end);
        
        % Max distance after entry (chattering)
        env.max_d_after = max(d(first_idx:end));
        
        % Mean distance in steady-state
        ss_idx = round(N*0.5);
        env.mean_d_ss = mean(d(ss_idx:end));
    end
end