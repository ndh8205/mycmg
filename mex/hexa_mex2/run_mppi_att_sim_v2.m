function results = run_mppi_att_sim_v2(mppi_params, params, t_end, verbose)
% run_mppi_att_sim_v2: MPPI v2 attitude control simulation (for tuning)
%
% Inputs:
%   mppi_params - MPPI parameter struct (v2 format)
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

ctrl_hz = 100;
ctrl_decimation = sim_hz / ctrl_hz;
dt_ctrl = 1 / ctrl_hz;

%% Process noise covariance
Q = zeros(9,9);

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
x0(1:3)   = [0; 0; -1];
x0(4:6)   = [0; 0; 0];
x0(7:10)  = q0;
x0(11:13) = [0; 0; 0];
x0(14:19) = omega_hover * ones(6,1);
x0(20:28) = zeros(9, 1);

%% Desired states
pos_des = [0; 0; -10];
q_des = [1; 0; 0; 0];  % Level attitude

%% Initialize MPPI state (v2 format)
mppi_state.u_seq = single(omega_hover * ones(6, mppi_params.N));
mppi_state.u_prev = single(omega_hover * ones(6, 1));
mppi_state.pos_des = single(pos_des);
mppi_state.q_des = single(q_des);
mppi_state.omega_motor_est = omega_hover * ones(6,1);

if ~isfield(mppi_params, 'K_pid')
    mppi_params.K_pid = 32;
    mppi_params.Kp_pos = [1; 1; 2];
    mppi_params.Kd_pos = [2; 2; 3];
    mppi_params.Kp_att = [8; 8; 6];
    mppi_params.Kd_att = [2; 2; 1.5];
    mppi_params.sigma_pid = 0.2;
end

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
diverge_reason = 'none';

for k = 1:N_sim-1
    x_k = X(:,k);
    
    % Check for divergence
    euler_k = Quat2Euler(x_k(7:10));
    alt_k = -x_k(3);  % positive up
    
    % NaN/Inf check
    if any(isnan(x_k)) || any(isinf(x_k))
        diverged = true;
        diverge_reason = 'nan_inf';
        diverge_time = t(k);
        X(:, k+1:end) = repmat(x_k, 1, N_sim-k);
        break;
    end
    
    % Flip check (roll/pitch > 70 deg)
    if max(abs(rad2deg(euler_k(1:2)))) > 70
        diverged = true;
        diverge_reason = 'flip';
        diverge_time = t(k);
        X(:, k+1:end) = repmat(x_k, 1, N_sim-k);
        break;
    end
    
    % Ground collision (altitude < 0.5m)
    if alt_k < 0.5
        diverged = true;
        diverge_reason = 'ground';
        diverge_time = t(k);
        X(:, k+1:end) = repmat(x_k, 1, N_sim-k);
        break;
    end
    
    % Altitude runaway (started at 10m, deviated > 15m)
    if alt_k > 25 || alt_k < 0
        diverged = true;
        diverge_reason = 'altitude_runaway';
        diverge_time = t(k);
        X(:, k+1:end) = repmat(x_k, 1, N_sim-k);
        break;
    end
    
    % XY drift too large (> 50m)
    if norm(x_k(1:2)) > 50
        diverged = true;
        diverge_reason = 'xy_drift';
        diverge_time = t(k);
        X(:, k+1:end) = repmat(x_k, 1, N_sim-k);
        break;
    end
    
    % MPPI control update
    if mod(k-1, ctrl_decimation) == 0
        % Update motor state estimate
        for i = 1:6
            if mppi_state.u_prev(i) >= mppi_state.omega_motor_est(i)
                tau_i = params.drone.motor.tau_up;
            else
                tau_i = params.drone.motor.tau_down;
            end
            mppi_state.omega_motor_est(i) = mppi_state.omega_motor_est(i) + ...
                dt_ctrl * (mppi_state.u_prev(i) - mppi_state.omega_motor_est(i)) / tau_i;
        end
        
        tic_mppi = tic;
        [u_opt, mppi_state] = hexa_mppi_controller_mex_v2(x_k, mppi_state, mppi_params, params);
        exec_time = toc(tic_mppi);
        
        u_current = u_opt;
        
        % Log diagnostics (with fallback defaults)
        if isfield(mppi_state, 'stats') && isfield(mppi_state.stats, 'effective_sample_size')
            ess_log(end+1) = mppi_state.stats.effective_sample_size;
        else
            ess_log(end+1) = mppi_params.K * 0.1;  % Default 10%
        end
        if isfield(mppi_state, 'stats') && isfield(mppi_state.stats, 'saturation_ratio')
            sat_log(end+1) = mppi_state.stats.saturation_ratio;
        else
            sat_log(end+1) = 0.1;  % Default
        end
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

%% Quaternion error (geodesic distance)
q_err_deg = zeros(1, N_sim);
for k = 1:N_sim
    q_k = X(7:10, k);
    q_dot = abs(q_des' * q_k);
    q_err_deg(k) = 2 * acos(min(q_dot, 1)) * 180/pi;
end

%% Compute metrics

% Settling time (2 deg threshold on quaternion error)
settling_threshold = 2;  % deg
settled_idx = find(q_err_deg < settling_threshold, 1, 'first');
if ~isempty(settled_idx) && all(q_err_deg(settled_idx:end) < settling_threshold)
    settling_time = t(settled_idx);
else
    settling_time = t_end;
end

% Steady-state evaluation (last 50%)
ss_start_idx = round(N_sim * 0.5);
euler_ss = euler(:, ss_start_idx:end);
q_err_ss = q_err_deg(ss_start_idx:end);
alt_ss = -X(3, ss_start_idx:end);
pos_xy_ss = X(1:2, ss_start_idx:end);

% RMSE attitude (geodesic) [deg]
rmse_att = sqrt(mean(q_err_ss.^2));

% RMSE individual axes [deg]
rmse_roll = sqrt(mean(rad2deg(euler_ss(1,:)).^2));
rmse_pitch = sqrt(mean(rad2deg(euler_ss(2,:)).^2));
rmse_yaw = sqrt(mean(rad2deg(euler_ss(3,:)).^2));

% RMSE altitude [m]
alt_des_val = -pos_des(3);
rmse_alt = sqrt(mean((alt_ss - alt_des_val).^2));

% Max attitude error (steady-state) [deg]
max_att_ss = max(q_err_ss);

% XY drift [m]
xy_drift_ss = max(vecnorm(pos_xy_ss));
xy_drift_final = norm(X(1:2,end));

% Oscillation metric (angular rate std in steady-state)
omega_ss = X(11:13, ss_start_idx:end);
oscillation = mean(std(rad2deg(omega_ss), 0, 2));

% Max overshoot
max_overshoot = max(q_err_deg);

% ESS statistics (handle empty)
if ~isempty(ess_log)
    ess_mean = mean(ess_log);
    ess_min = min(ess_log);
else
    ess_mean = mppi_params.K * 0.1;
    ess_min = mppi_params.K * 0.1;
end

% Saturation statistics (handle empty)
if ~isempty(sat_log)
    sat_mean = mean(sat_log);
    sat_max = max(sat_log);
else
    sat_mean = 0.1;
    sat_max = 0.1;
end

% Execution time (handle empty)
if ~isempty(exec_time_log)
    exec_mean = mean(exec_time_log);
else
    exec_mean = 0.01;
end

%% Pack results
results.diverged = diverged;
results.diverge_time = diverge_time;
results.diverge_reason = diverge_reason;

% Safety metrics (전체 시뮬레이션)
euler_all = euler(:, 1:min(size(euler,2), find(~all(euler==0,1), 1, 'last')));
if isempty(euler_all)
    euler_all = euler;
end
results.max_tilt = max(abs(rad2deg(euler_all(1:2, :))), [], 'all');
results.min_alt = min(-X(3, X(3,:) ~= 0));
results.max_xy_drift = max(vecnorm(X(1:2, :)));

results.settling_time = settling_time;
results.max_overshoot = max_overshoot;

% Steady-state metrics
results.rmse_att = rmse_att;
results.rmse_roll = rmse_roll;
results.rmse_pitch = rmse_pitch;
results.rmse_yaw = rmse_yaw;
results.rmse_alt = rmse_alt;
results.max_att_ss = max_att_ss;
results.xy_drift_ss = xy_drift_ss;
results.xy_drift_final = xy_drift_final;
results.oscillation = oscillation;

% ESS and saturation
results.ess_mean = ess_mean;
results.ess_min = ess_min;
results.sat_mean = sat_mean;
results.sat_max = sat_max;
results.exec_mean = exec_mean;

% Raw data
results.t = t;
results.euler = euler;
results.q_err_deg = q_err_deg;
results.X = X;

if verbose
    if diverged
        fprintf('*** DIVERGED at t=%.2fs ***\n', diverge_time);
    end
    fprintf('Settling time: %.2f s\n', settling_time);
    fprintf('RMSE(ss): att=%.3f deg, alt=%.4f m\n', rmse_att, rmse_alt);
    fprintf('Max(ss): att=%.2f deg, drift=%.3f m\n', max_att_ss, xy_drift_ss);
    fprintf('Oscillation: %.2f deg/s, Overshoot: %.2f deg\n', oscillation, max_overshoot);
    fprintf('ESS: %.1f (min %.1f), Sat: %.1f%% (max %.1f%%)\n', ...
        ess_mean, ess_min, sat_mean*100, sat_max*100);
end

end