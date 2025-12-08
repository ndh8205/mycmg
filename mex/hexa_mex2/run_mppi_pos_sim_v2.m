function results = run_mppi_pos_sim_v2(mppi_params, params, t_end, verbose)
% run_mppi_pos_sim_v2: MPPI v2 position control simulation (for tuning)
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

%% Waypoints [x, y, z] in NED
waypoints = [
    0,   0,  -10;   % WP1: Hover at start
    10,  0,  -10;   % WP2: Forward
    10,  10, -10;   % WP3: Right
    0,   10, -10;   % WP4: Back
    0,   0,  -10;   % WP5: Return
];
n_wp = size(waypoints, 1);
wp_idx = 1;
wp_threshold = 1.0;  % [m] arrival threshold
yaw_des = 0;

%% Initial state
m = params.drone.body.m;
g = params.env.g;
k_T = params.drone.motor.k_T;
n_motor = params.drone.body.n_motor;

omega_hover = sqrt(m * g / (n_motor * k_T));

x0 = zeros(28, 1);
x0(1:3)   = [0; 0; -10];              % Start at WP1
x0(4:6)   = [0; 0; 0];
x0(7:10)  = [1; 0; 0; 0];
x0(11:13) = [0; 0; 0];
x0(14:19) = omega_hover * ones(6,1);
x0(20:28) = zeros(9, 1);

%% Initialize MPPI state (v2 format)
mppi_state.u_seq = single(omega_hover * ones(6, mppi_params.N));
mppi_state.u_prev = single(omega_hover * ones(6, 1));
mppi_state.pos_des = single(waypoints(1,:)');
mppi_state.q_des = single([1; 0; 0; 0]);  % Level attitude
mppi_state.omega_motor_est = omega_hover * ones(6,1);

%% Data storage
X = zeros(28, N_sim);
U = zeros(6, N_sim);
POS_DES = zeros(3, N_sim);
X(:,1) = x0;
POS_DES(:,1) = waypoints(1,:)';
u_current = omega_hover * ones(6,1);

% Waypoint tracking
wp_reached_time = zeros(n_wp, 1);
wp_reached_time(1) = 0;  % Start at WP1

% Diagnostics
ess_log = [];
sat_log = [];
exec_time_log = [];

%% Simulation loop
diverged = false;
diverge_time = t_end;

for k = 1:N_sim-1
    x_k = X(:,k);
    
    % Extract states
    pos_ned = x_k(1:3);
    
    % Check for divergence
    euler_k = Quat2Euler(x_k(7:10));
    if any(isnan(x_k)) || any(isinf(x_k)) || ...
       max(abs(rad2deg(euler_k(1:2)))) > 89 || ...
       -x_k(3) < -50 || -x_k(3) > 500 || ...
       norm(x_k(1:2)) > 200
        diverged = true;
        diverge_time = t(k);
        X(:, k+1:end) = repmat(x_k, 1, N_sim-k);
        break;
    end
    
    % Waypoint arrival check
    pos_des = waypoints(wp_idx, :)';
    dist_to_wp = norm(pos_ned - pos_des);
    if dist_to_wp < wp_threshold && wp_idx < n_wp
        wp_idx = wp_idx + 1;
        wp_reached_time(wp_idx) = t(k);
        if verbose
            fprintf('  t=%.1fs: Reached WP%d\n', t(k), wp_idx-1);
        end
    end
    pos_des = waypoints(wp_idx, :)';
    POS_DES(:,k) = pos_des;
    
    % MPPI control update
    if mod(k-1, ctrl_decimation) == 0
        % Update desired position
        mppi_state.pos_des = single(pos_des);
        
        % Update desired quaternion (level with yaw)
        q_des = GetQUAT(yaw_des, 0, 0);
        mppi_state.q_des = single(q_des(:));
        
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
        
        % Shift control sequence
        mppi_state.u_seq(:, 1:end-1) = mppi_state.u_seq(:, 2:end);
        
        tic_mppi = tic;
        [u_opt, mppi_state] = hexa_mppi_controller_mex_v2(x_k, mppi_state, mppi_params, params);
        exec_time = toc(tic_mppi);
        
        u_current = u_opt;
        
        ess_log(end+1) = mppi_state.stats.effective_sample_size;
        sat_log(end+1) = mppi_state.stats.saturation_ratio;
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
POS_DES(:,end) = POS_DES(:,end-1);

%% Extract Euler angles
euler = zeros(3, N_sim);
for k = 1:N_sim
    euler(:,k) = Quat2Euler(X(7:10,k));
end

%% Position error
pos_err = X(1:3,:) - POS_DES;
pos_err_norm = vecnorm(pos_err);

%% Compute metrics

% Waypoints reached
wp_reached = wp_idx;

% Total path completion time
if wp_reached == n_wp
    path_complete_time = wp_reached_time(n_wp);
else
    path_complete_time = t_end;  % Did not complete
end

% Average position tracking error
rmse_pos = sqrt(mean(pos_err_norm.^2));
rmse_pos_xy = sqrt(mean(pos_err(1,:).^2 + pos_err(2,:).^2));
rmse_pos_z = sqrt(mean(pos_err(3,:).^2));

% Max position error
max_pos_err = max(pos_err_norm);

% Final position error
final_pos_err = pos_err_norm(end);

% Altitude tracking
alt_des = -POS_DES(3,:);
alt_actual = -X(3,:);
rmse_alt = sqrt(mean((alt_actual - alt_des).^2));

% Attitude metrics (steady-state: last 30%)
ss_start_idx = round(N_sim * 0.7);
euler_ss = euler(:, ss_start_idx:end);
max_att = max(abs(rad2deg(euler_ss(:))));
rmse_att = sqrt(mean(rad2deg(euler_ss(1,:)).^2 + rad2deg(euler_ss(2,:)).^2));

% Oscillation (angular rate std)
omega_all = X(11:13, :);
oscillation = mean(std(rad2deg(omega_all), 0, 2));

% ESS and saturation
ess_mean = mean(ess_log);
ess_min = min(ess_log);
sat_mean = mean(sat_log);
sat_max = max(sat_log);
exec_mean = mean(exec_time_log);

%% Pack results
results.diverged = diverged;
results.diverge_time = diverge_time;

% Waypoint metrics
results.wp_reached = wp_reached;
results.wp_total = n_wp;
results.path_complete_time = path_complete_time;
results.wp_reached_time = wp_reached_time;

% Position metrics
results.rmse_pos = rmse_pos;
results.rmse_pos_xy = rmse_pos_xy;
results.rmse_pos_z = rmse_pos_z;
results.rmse_alt = rmse_alt;
results.max_pos_err = max_pos_err;
results.final_pos_err = final_pos_err;

% Attitude metrics
results.max_att = max_att;
results.rmse_att = rmse_att;
results.oscillation = oscillation;

% ESS and saturation
results.ess_mean = ess_mean;
results.ess_min = ess_min;
results.sat_mean = sat_mean;
results.sat_max = sat_max;
results.exec_mean = exec_mean;

% Raw data
results.t = t;
results.X = X;
results.POS_DES = POS_DES;
results.euler = euler;
results.pos_err_norm = pos_err_norm;

if verbose
    if diverged
        fprintf('*** DIVERGED at t=%.2fs ***\n', diverge_time);
    end
    fprintf('Waypoints: %d/%d\n', wp_reached, n_wp);
    fprintf('Path time: %.2f s\n', path_complete_time);
    fprintf('RMSE: pos=%.3f m, alt=%.3f m, att=%.2f deg\n', rmse_pos, rmse_alt, rmse_att);
    fprintf('Max: pos_err=%.3f m, att=%.2f deg\n', max_pos_err, max_att);
    fprintf('ESS: %.1f (min %.1f), Sat: %.1f%%\n', ess_mean, ess_min, sat_mean*100);
end

end