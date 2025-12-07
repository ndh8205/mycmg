%% main_mppi_hexa.m - Hexarotor MPPI Position Control Test
% Model Predictive Path Integral Control (Williams et al. 2017)
clear; clc; close all;

addpath(genpath('math'));
addpath(genpath('model'));
addpath(genpath('ctrl'));
addpath(genpath('disturbance'));

%% Parameters (Hexarotor)
params = params_init('hexa');

%% Disturbance settings
dist_preset = 'level2';
[params, dist_state] = dist_init(params, dist_preset);
if params.dist.uncertainty.enable
    params_true = apply_uncertainty(params);
else
    params_true = params;
end

%% Simulation settings
sim_hz = 1000;          % Simulation frequency [Hz]
ctrl_hz = 50;           % MPPI control frequency [Hz]
dt_sim = 1/sim_hz;
dt_ctrl = 1/ctrl_hz;
ctrl_decimation = sim_hz / ctrl_hz;

t_end = 30;
t = 0:dt_sim:t_end;
N_sim = length(t);

omega_bar2RPM = 60 / (2 * pi);

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
wp_threshold = 0.5;  % [m] arrival threshold
yaw_des = 0;

%% MPPI Parameters (Optimized)
mppi_params.K = 512;            % Rollouts (2048→512: 4x speedup)
mppi_params.N = 25;             % Horizon (50→25: 2x speedup)
mppi_params.dt = dt_ctrl;       % Rollout timestep [s]
mppi_params.lambda = 0.1;       % Temperature (1.0→0.1: exploitation)
mppi_params.nu = 100;           % Exploration variance

% Control noise [rad/s] (50→15: reduce flip)
mppi_params.sigma = 15 * ones(6, 1);

% Cost weights
mppi_params.w_pos = 10.0;       % Position error
mppi_params.w_vel = 1.0;        % Velocity
mppi_params.w_att = 20000.0;      % Attitude (50→200: prevent flip)
mppi_params.w_yaw = 500.0;        % Yaw error
mppi_params.w_omega = 1000.0;      % Angular velocity (0.1→1.0)
mppi_params.w_terminal = 5000.0;  % Terminal cost

% Control cost (0.001→0.01: smoother)
mppi_params.R = 0.001 * ones(6, 1);

%% Initial state (28x1 for Hexa)
m = params.drone.body.m;
g = params.env.g;
k_T = params.drone.motor.k_T;
n_motor = params.drone.body.n_motor;

omega_hover = sqrt(m * g / (n_motor * k_T));
fprintf('Hover motor speed: %.2f RPM\n', omega_hover * omega_bar2RPM);

x0 = zeros(28, 1);
x0(1:3)   = [0; 0; -0.01];            % position (NED)
x0(4:6)   = [0; 0; 0];                % velocity (body)
x0(7:10)  = [1; 0; 0; 0];             % quaternion (level)
x0(11:13) = [0; 0; 0];                % angular velocity
x0(14:19) = omega_hover * ones(6,1);  % motor speed (6 motors)
x0(20:28) = zeros(9, 1);              % biases

%% MPPI State initialization
mppi_state.u_seq = omega_hover * ones(6, mppi_params.N);  % Initial control sequence
mppi_state.pos_des = waypoints(1, :)';
mppi_state.yaw_des = yaw_des;

%% Data storage
X = zeros(28, N_sim);
U = zeros(6, N_sim);
POS_DES = zeros(3, N_sim);
TAU_DIST = zeros(3, N_sim);
F_DIST = zeros(3, N_sim);
MPPI_TIME = zeros(1, ceil(N_sim/ctrl_decimation));
X(:,1) = x0;
POS_DES(:,1) = waypoints(1,:)';

%% GPU Warm-up
fprintf('GPU warm-up...\n');
[~, mppi_state] = mppi_controller(x0, mppi_state, mppi_params, params);
fprintf('GPU warm-up complete.\n');

%% Simulation loop
fprintf('Running MPPI simulation (dist: %s)...\n', dist_preset);
fprintf('K=%d, N=%d, ctrl_hz=%d\n', mppi_params.K, mppi_params.N, ctrl_hz);

u_current = omega_hover * ones(6, 1);
mppi_iter = 0;

tic;
for k = 1:N_sim-1
    x_k = X(:,k);
    
    % Extract states
    pos_ned = x_k(1:3);
    vel_b = x_k(4:6);
    q = x_k(7:10);
    
    % Velocity in NED
    R_b2n = GetDCM_QUAT(q);
    
    % Waypoint arrival check
    pos_des = waypoints(wp_idx, :)';
    dist_to_wp = norm(pos_ned - pos_des);
    if dist_to_wp < wp_threshold && wp_idx < n_wp
        wp_idx = wp_idx + 1;
        fprintf('  t=%.1fs: Reached WP%d, heading to WP%d\n', t(k), wp_idx-1, wp_idx);
    end
    pos_des = waypoints(wp_idx, :)';
    POS_DES(:,k) = pos_des;
    
    % MPPI control (at control frequency)
    if mod(k-1, ctrl_decimation) == 0
        mppi_iter = mppi_iter + 1;
        
        % Update desired state
        mppi_state.pos_des = pos_des;
        mppi_state.yaw_des = yaw_des;
        
        % Run MPPI
        tic_mppi = tic;
        [u_current, mppi_state] = mppi_controller(x_k, mppi_state, mppi_params, params);
        MPPI_TIME(mppi_iter) = toc(tic_mppi);
    end
    
    % Apply control
    u = u_current;
    
    % Saturate motor commands
    omega_max = params.drone.motor.omega_b_max;
    omega_min = params.drone.motor.omega_b_min;
    u = max(min(u, omega_max), omega_min);
    U(:,k) = u;
    
    % Log disturbance
    if params.dist.enable
        [tau_d, ~] = dist_torque(t(k), params, dist_state);
        [F_d, ~] = dist_wind(vel_b, R_b2n, t(k), dt_sim, params, dist_state);
        TAU_DIST(:,k) = tau_d;
        F_DIST(:,k) = F_d;
    end
    
    % Integration (use params_true for dynamics)
    [x_next, ~, ~, ~, ~, ~, dist_state] = srk4(@drone_dynamics, x_k, u, Q, dt_sim, params_true, k, dt_sim, t(k), dist_state);
    
    % Normalize quaternion and ensure continuity
    x_next(7:10) = x_next(7:10) / norm(x_next(7:10));
    x_next(7:10) = EnsQuatCont(x_next(7:10), x_k(7:10));
    
    X(:,k+1) = x_next;
    
    % Progress
    if mod(k, 5000) == 0
        fprintf('  Progress: %.1f%%\n', 100*k/N_sim);
    end
end
total_time = toc;
U(:,end) = U(:,end-1);
POS_DES(:,end) = POS_DES(:,end-1);

fprintf('Simulation complete. Total time: %.2f s (%.2fx realtime)\n', ...
    total_time, t_end/total_time);

%% Extract Euler angles
euler = zeros(3, N_sim);
for k = 1:N_sim
    euler(:,k) = Quat2Euler(X(7:10,k));
end

%% MPPI Timing statistics
mppi_times_valid = MPPI_TIME(1:mppi_iter);
fprintf('\n=== MPPI Timing Statistics ===\n');
fprintf('Mean iteration time: %.2f ms\n', 1000*mean(mppi_times_valid));
fprintf('Max iteration time:  %.2f ms\n', 1000*max(mppi_times_valid));
fprintf('Min iteration time:  %.2f ms\n', 1000*min(mppi_times_valid));
fprintf('Control period:      %.2f ms\n', 1000*dt_ctrl);
fprintf('Real-time capable:   %s\n', ...
    char(ternary(mean(mppi_times_valid) < dt_ctrl, "YES", "NO")));

%% Plot results
figure('Position', [100 100 1400 900], 'Name', 'MPPI Position Control');

% X position
subplot(3,3,1);
plot(t, X(1,:), 'b', 'LineWidth', 1.5); hold on;
plot(t, POS_DES(1,:), 'r--', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('X [m]');
title('X Position'); grid on;
legend('Actual', 'Desired');

% Y position
subplot(3,3,2);
plot(t, X(2,:), 'b', 'LineWidth', 1.5); hold on;
plot(t, POS_DES(2,:), 'r--', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('Y [m]');
title('Y Position'); grid on;
legend('Actual', 'Desired');

% Z position (altitude)
subplot(3,3,3);
plot(t, -X(3,:), 'b', 'LineWidth', 1.5); hold on;
plot(t, -POS_DES(3,:), 'r--', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('Altitude [m]');
title('Altitude'); grid on;
legend('Actual', 'Desired');

% Euler angles
subplot(3,3,4);
plot(t, rad2deg(euler));
xlabel('Time [s]'); ylabel('Angle [deg]');
legend('\phi','\theta','\psi');
title('Euler Angles'); grid on;

% Motor speed (actual)
subplot(3,3,5);
plot(t, X(14:19,:) * omega_bar2RPM);
xlabel('Time [s]'); ylabel('Motor Speed [RPM]');
legend('M1','M2','M3','M4','M5','M6');
title('Motor Speed (Actual)'); grid on;

% 3D trajectory
subplot(3,3,6);
plot3(X(1,:), X(2,:), -X(3,:), 'b-', 'LineWidth', 1.5); hold on;
plot3(waypoints(:,1), waypoints(:,2), -waypoints(:,3), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
plot3(X(1,1), X(2,1), -X(3,1), 'gs', 'MarkerSize', 10, 'LineWidth', 2);
xlabel('X [m]'); ylabel('Y [m]'); zlabel('Alt [m]');
title('3D Trajectory'); grid on; axis equal;
legend('Path', 'Waypoints', 'Start');
view(45, 30);

% MPPI timing
subplot(3,3,7);
plot(1:mppi_iter, 1000*mppi_times_valid, 'b-');
hold on;
yline(1000*dt_ctrl, 'r--', 'LineWidth', 1.5);
xlabel('Iteration'); ylabel('Time [ms]');
title('MPPI Computation Time');
legend('Actual', 'Real-time limit');
grid on;

% Control commands
subplot(3,3,8);
plot(t, U * omega_bar2RPM);
xlabel('Time [s]'); ylabel('Motor Command [RPM]');
legend('M1','M2','M3','M4','M5','M6');
title('Motor Commands'); grid on;

% Position error
subplot(3,3,9);
pos_err = X(1:3,:) - POS_DES;
plot(t, vecnorm(pos_err), 'k', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('|Error| [m]');
title('Position Error'); grid on;

sgtitle(sprintf('MPPI Control (K=%d, N=%d, Dist: %s)', ...
    mppi_params.K, mppi_params.N, dist_preset));

%% Print summary
fprintf('\n=== MPPI Position Control Summary ===\n');
fprintf('Preset: %s\n', dist_preset);
fprintf('Final position error: %.3f m\n', norm(X(1:3,end) - POS_DES(:,end)));
fprintf('Max position error: %.3f m\n', max(vecnorm(pos_err)));
fprintf('Max attitude: %.2f deg\n', max(abs(rad2deg(euler(:)))));
fprintf('Waypoints reached: %d/%d\n', wp_idx, n_wp);
fprintf('=====================================\n');

% %% Export to CSV
% data = [t', X(1:3,:)', POS_DES', rad2deg(euler)', X(11:13,:)', U', X(14:19,:)', ...
%         TAU_DIST', F_DIST'];
% header = {'t','pos_x','pos_y','pos_z','pos_des_x','pos_des_y','pos_des_z',...
%           'euler_roll','euler_pitch','euler_yaw','omega_p','omega_q','omega_r',...
%           'motor_cmd_1','motor_cmd_2','motor_cmd_3','motor_cmd_4','motor_cmd_5','motor_cmd_6',...
%           'motor_actual_1','motor_actual_2','motor_actual_3','motor_actual_4','motor_actual_5','motor_actual_6',...
%           'tau_dist_x','tau_dist_y','tau_dist_z','F_dist_x','F_dist_y','F_dist_z'};
% filename = sprintf('mppi_%s.csv', dist_preset);
% fid = fopen(filename, 'w');
% fprintf(fid, '%s,', header{1:end-1}); fprintf(fid, '%s\n', header{end});
% fclose(fid);
% dlmwrite(filename, data, '-append', 'precision', '%.6f');
% fprintf('Data saved to: %s\n', filename);

%% Helper function
function out = ternary(cond, true_val, false_val)
    if cond
        out = true_val;
    else
        out = false_val;
    end
end