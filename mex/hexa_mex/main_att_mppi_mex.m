%% main_att_mppi_mex.m - Hexarotor MPPI Attitude & Altitude Control
% CUDA MEX version with motor dynamics
clear; clc; close all;

addpath(genpath('math'));
addpath(genpath('model'));
addpath(genpath('ctrl'));
addpath(genpath('disturbance'));

%% Check MEX file
if ~exist('hexa_mppi_mex', 'file')
    error('hexa_mppi_mex not found. Run: mexcuda hexa_mppi_mex.cu -lcurand');
end

%% Parameters (Hexarotor)
params = params_init('hexa');

%% Disturbance settings
dist_preset = 'nominal';
% dist_preset = 'level1';
% dist_preset = 'level2';

[params, dist_state] = dist_init(params, dist_preset);
if params.dist.uncertainty.enable
    params_true = apply_uncertainty(params);
else
    params_true = params;
end

%% Simulation settings
sim_hz = 1000;
dt_sim = 1/sim_hz;
t_end = 10;
t = 0:dt_sim:t_end;
N_sim = length(t);

ctrl_hz = 100;  % MPPI control rate
ctrl_decimation = sim_hz / ctrl_hz;
dt_ctrl = 1 / ctrl_hz;

omega_bar2RPM = 60 / (2 * pi);

%% Process noise covariance
Q = zeros(9,9);

% mppi_params.K = 4096;
% mppi_params.N = 100;
% mppi_params.dt = dt_ctrl;
% mppi_params.lambda = 807.97;
% mppi_params.nu = 10.0;
% mppi_params.sigma = 35.1773;  % 335.9 RPM
% mppi_params.w_pos_xy = 173.47;
% mppi_params.w_pos_z = 105.31;
% mppi_params.w_vel_xy = 1500.00;
% mppi_params.w_vel_z = 395.21;
% mppi_params.w_att = 100.00;
% mppi_params.w_yaw = 2348.33;
% mppi_params.w_omega_rp = 634.56;
% mppi_params.w_omega_yaw = 1000.00;
% mppi_params.w_terminal = 118.47;
% mppi_params.R = 1.00e-05;
% 
% mppi_params.K = 4096;
% mppi_params.N = 100;
% mppi_params.dt = dt_ctrl;
% mppi_params.lambda = 717.64;
% mppi_params.nu = 10.0;
% mppi_params.sigma = 38.1090;  % 363.9 RPM
% mppi_params.w_pos_xy = 181.10;
% mppi_params.w_pos_z = 108.29;
% mppi_params.w_vel_xy = 1319.31;
% mppi_params.w_vel_z = 386.42;
% mppi_params.w_att = 37.35;
% mppi_params.w_yaw = 2496.58;
% mppi_params.w_omega_rp = 633.88;
% mppi_params.w_omega_yaw = 1011.16;
% mppi_params.w_terminal = 116.07;
% mppi_params.R = 2.50e-06;

% mppi_params.K = 4096;
% mppi_params.N = 100;
% mppi_params.dt = dt_ctrl;
% mppi_params.lambda = 2471.72;
% mppi_params.nu = 10.0;
% mppi_params.sigma = 27.6990;  % 264.5 RPM
% mppi_params.w_pos_xy = 48.55;
% mppi_params.w_pos_z = 125.15;
% mppi_params.w_vel_xy = 1194.60;
% mppi_params.w_vel_z = 514.56;
% mppi_params.w_att = 2760.72;
% mppi_params.w_yaw = 1413.94;
% mppi_params.w_omega_rp = 735.92;
% mppi_params.w_omega_yaw = 327.62;
% mppi_params.w_terminal = 149.04;
% mppi_params.R = 4.81e-05;

mppi_params.K = 4096;
mppi_params.N = 100;
mppi_params.dt = dt_ctrl;
mppi_params.lambda = 8.7310;
mppi_params.nu = 10.0;
mppi_params.sigma = 15.1658;  % 144.8 RPM
mppi_params.w_pos_xy = 0.0037;
mppi_params.w_pos_z = 100.2281;
mppi_params.w_vel_xy = 0.2683;
mppi_params.w_vel_z = 1529.8645;
mppi_params.w_att = 142.9748;
mppi_params.w_yaw = 3776.0578;
mppi_params.w_omega_rp = 0.0075;
mppi_params.w_omega_yaw = 360.6177;
mppi_params.w_terminal = 1.2051;
mppi_params.R = 1.81e-08;

%% Drone parameters for MEX (single precision struct)
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
drone_params.tau_up = single(params.drone.motor.tau_up);
drone_params.tau_down = single(params.drone.motor.tau_down);

%% Initial state (28x1 for hexa)
m = params.drone.body.m;
g = params.env.g;
k_T = params.drone.motor.k_T;
n_motor = params.drone.body.n_motor;

omega_hover = sqrt(m * g / (n_motor * k_T));
fprintf('Hover motor speed: %.2f RPM (%.2f rad/s)\n', omega_hover * omega_bar2RPM, omega_hover);

% Initial euler: roll=20, pitch=15, yaw=10 deg
euler0 = deg2rad([20; 15; 10]);
q0 = GetQUAT(euler0(3), euler0(2), euler0(1));
q0 = q0(:);

x0 = zeros(28, 1);
x0(1:3)   = [0; 0; -1];              % position (NED), 10m altitude
x0(4:6)   = [0; 0; 0];                % velocity (body)
x0(7:10)  = q0;                       % quaternion
x0(11:13) = [0; 0; 0];                % angular velocity
x0(14:19) = omega_hover * ones(6,1);  % motor speed (6 motors)
x0(20:28) = zeros(9, 1);              % biases

%% Desired states
pos_des = [0; 0; -10];
yaw_des = 0;

%% Initialize MPPI state
mppi_state.u_seq = single(omega_hover * ones(6, mppi_params.N));
mppi_state.pos_des = single(pos_des);
mppi_state.yaw_des = single(yaw_des);

%% Motor state estimator initialization
omega_motor_est = omega_hover * ones(6,1);
u_prev = omega_hover * ones(6,1);

%% Data storage
X = zeros(28, N_sim);
U = zeros(6, N_sim);
TAU_DIST = zeros(3, N_sim);
F_DIST = zeros(3, N_sim);

% MPPI diagnostics
N_ctrl = ceil(N_sim / ctrl_decimation);
MPPI_STATS.min_cost = zeros(1, N_ctrl);
MPPI_STATS.avg_cost = zeros(1, N_ctrl);
MPPI_STATS.cost_breakdown = zeros(6, N_ctrl);
MPPI_STATS.sat_ratio = zeros(1, N_ctrl);
MPPI_STATS.ess = zeros(1, N_ctrl);
MPPI_STATS.time = zeros(1, N_ctrl);
MPPI_STATS.exec_time = zeros(1, N_ctrl);

X(:,1) = x0;
u_current = omega_hover * ones(6,1);
ctrl_idx = 0;

%% Simulation loop
fprintf('Running MPPI attitude control (dist: %s)...\n', dist_preset);
fprintf('K=%d, N=%d, sigma=%.1f, lambda=%.1f\n', ...
    mppi_params.K, mppi_params.N, mppi_params.sigma, mppi_params.lambda);

tic_total = tic;
for k = 1:N_sim-1
    x_k = X(:,k);
    
    % MPPI control update at control rate
    if mod(k-1, ctrl_decimation) == 0
        ctrl_idx = ctrl_idx + 1;
        
        % Update motor state estimate (1st order dynamics)
        for i = 1:6
            if u_prev(i) >= omega_motor_est(i)
                tau_i = params.drone.motor.tau_up;
            else
                tau_i = params.drone.motor.tau_down;
            end
            omega_motor_est(i) = omega_motor_est(i) + dt_ctrl * (u_prev(i) - omega_motor_est(i)) / tau_i;
        end
        
        % Prepare state for MEX (19x1, single precision)
        % [pos(3), vel(3), quat(4), omega(3), omega_motor(6)]
        x_mppi = single([x_k(1:3); x_k(4:6); x_k(7:10); x_k(11:13); omega_motor_est]);
        
        % Shift control sequence (warm start)
        mppi_state.u_seq(:, 1:end-1) = mppi_state.u_seq(:, 2:end);
        
        % Call MPPI MEX
        tic_mppi = tic;
        [u_opt, u_seq_new, stats] = hexa_mppi_mex(...
            x_mppi, mppi_state.u_seq, mppi_state.pos_des, mppi_state.yaw_des, ...
            drone_params, mppi_params);
        exec_time = toc(tic_mppi);
        
        % Update state
        u_current = double(u_opt);
        mppi_state.u_seq = u_seq_new;
        u_prev = u_current;
        
        % Store diagnostics
        MPPI_STATS.min_cost(ctrl_idx) = stats.min_cost;
        MPPI_STATS.avg_cost(ctrl_idx) = stats.avg_cost;
        MPPI_STATS.cost_breakdown(:, ctrl_idx) = stats.cost_breakdown;
        MPPI_STATS.sat_ratio(ctrl_idx) = stats.saturation_ratio;
        MPPI_STATS.ess(ctrl_idx) = stats.effective_sample_size;
        MPPI_STATS.time(ctrl_idx) = t(k);
        MPPI_STATS.exec_time(ctrl_idx) = exec_time;
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
        vel_b = x_k(4:6);
        q = x_k(7:10);
        R_b2n = GetDCM_QUAT(q);
        [tau_d, ~] = dist_torque(t(k), params, dist_state);
        [F_d, ~] = dist_wind(vel_b, R_b2n, t(k), dt_sim, params, dist_state);
        TAU_DIST(:,k) = tau_d;
        F_DIST(:,k) = F_d;
    end
    
    % Integration
    [x_next, ~, ~, ~, ~, ~, dist_state] = srk4(@drone_dynamics, x_k, u, Q, dt_sim, params_true, k, dt_sim, t(k), dist_state);
    
    % Normalize quaternion and ensure continuity
    x_next(7:10) = x_next(7:10) / norm(x_next(7:10));
    x_next(7:10) = EnsQuatCont(x_next(7:10), x_k(7:10));
    
    X(:,k+1) = x_next;
end
U(:,end) = U(:,end-1);
total_time = toc(tic_total);

fprintf('Simulation complete. Total: %.2fs, Real-time factor: %.1fx\n', ...
    total_time, t_end/total_time);
fprintf('MPPI avg exec time: %.3f ms (%.1f Hz)\n', ...
    mean(MPPI_STATS.exec_time(1:ctrl_idx))*1000, 1/mean(MPPI_STATS.exec_time(1:ctrl_idx)));

%% Extract Euler angles
euler = zeros(3, N_sim);
for k = 1:N_sim
    euler(:,k) = Quat2Euler(X(7:10,k));
end

%% Plot results
figure('Position', [100 100 1400 900], 'Name', 'MPPI Attitude & Altitude Control');

% Altitude
subplot(3,3,1);
plot(t, -X(3,:), 'b', 'LineWidth', 1.5);
hold on;
yline(-pos_des(3), 'r--', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('Altitude [m]');
title('Altitude'); grid on;
legend('Actual', 'Desired');

% Euler angles
subplot(3,3,2);
plot(t, rad2deg(euler));
xlabel('Time [s]'); ylabel('Angle [deg]');
legend('\phi','\theta','\psi'); 
title('Euler Angles'); grid on;
yline(0, 'k--');

% Angular velocity
subplot(3,3,3);
plot(t, rad2deg(X(11:13,:)));
xlabel('Time [s]'); ylabel('Angular Rate [deg/s]');
legend('p','q','r'); 
title('Angular Velocity (Body)'); grid on;

% Motor commands
subplot(3,3,4);
plot(t, U * omega_bar2RPM);
xlabel('Time [s]'); ylabel('Motor Speed [RPM]');
legend('M1','M2','M3','M4','M5','M6');
title('Motor Commands'); grid on;

% Motor speed (actual)
subplot(3,3,5);
plot(t, X(14:19,:) * omega_bar2RPM);
xlabel('Time [s]'); ylabel('Motor Speed [RPM]');
legend('M1','M2','M3','M4','M5','M6');
title('Motor Speed (Actual)'); grid on;

% Horizontal position
subplot(3,3,6);
plot(t, X(1:2,:));
xlabel('Time [s]'); ylabel('Position [m]');
legend('x','y');
title('Horizontal Position (NED)'); grid on;

% Torque disturbance
subplot(3,3,7);
plot(t, TAU_DIST');
xlabel('Time [s]'); ylabel('Torque [Nm]');
legend('\tau_x','\tau_y','\tau_z');
title('Torque Disturbance'); grid on;

% Wind force
subplot(3,3,8);
plot(t, F_DIST');
xlabel('Time [s]'); ylabel('Force [N]');
legend('F_x','F_y','F_z');
title('Wind Force (Body)'); grid on;

% Horizontal drift
subplot(3,3,9);
plot(t, vecnorm(X(1:2,:)), 'b', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('|XY Error| [m]');
title('Horizontal Drift'); grid on;

sgtitle(sprintf('Hexarotor MPPI Control (Dist: %s, K=%d)', dist_preset, mppi_params.K));

%% Figure 2: MPPI Diagnostics
figure('Position', [150 100 1400 700], 'Name', 'MPPI Diagnostics');

t_ctrl = MPPI_STATS.time(1:ctrl_idx);

% Cost breakdown
subplot(2,3,1);
cost_labels = {'Position', 'Velocity', 'Attitude', 'Yaw', 'Omega', 'Control'};
area(t_ctrl, MPPI_STATS.cost_breakdown(:, 1:ctrl_idx)');
xlabel('Time [s]'); ylabel('Cost');
title('Cost Breakdown (Stacked)');
legend(cost_labels, 'Location', 'best');
grid on;

% Cost over time
subplot(2,3,2);
semilogy(t_ctrl, MPPI_STATS.min_cost(1:ctrl_idx), 'b-', 'LineWidth', 1.5);
hold on;
semilogy(t_ctrl, MPPI_STATS.avg_cost(1:ctrl_idx), 'r--', 'LineWidth', 1);
xlabel('Time [s]'); ylabel('Cost (log)');
title('Min/Avg Cost');
legend('Min', 'Avg');
grid on;

% Saturation ratio
subplot(2,3,3);
plot(t_ctrl, MPPI_STATS.sat_ratio(1:ctrl_idx) * 100, 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('Saturation [%]');
title('Control Saturation Ratio');
ylim([0 100]);
grid on;

% Effective Sample Size
subplot(2,3,4);
plot(t_ctrl, MPPI_STATS.ess(1:ctrl_idx), 'LineWidth', 1.5);
hold on;
yline(mppi_params.K * 0.1, 'r--', 'LineWidth', 1);
xlabel('Time [s]'); ylabel('ESS');
title(sprintf('Effective Sample Size (K=%d)', mppi_params.K));
legend('ESS', '10% K');
grid on;

% Execution time
subplot(2,3,5);
plot(t_ctrl, MPPI_STATS.exec_time(1:ctrl_idx) * 1000, 'LineWidth', 1.5);
hold on;
yline(dt_ctrl * 1000, 'r--', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('Exec Time [ms]');
title('MPPI Execution Time');
legend('Actual', sprintf('Budget (%.1fms)', dt_ctrl*1000));
grid on;

% Phase portrait (Roll)
subplot(2,3,6);
plot(rad2deg(euler(1,:)), rad2deg(X(11,:)), 'b-', 'LineWidth', 0.5);
hold on;
plot(rad2deg(euler(1,1)), rad2deg(X(11,1)), 'go', 'MarkerSize', 10, 'LineWidth', 2);
plot(0, 0, 'r+', 'MarkerSize', 15, 'LineWidth', 2);
xlabel('Roll [deg]'); ylabel('Roll Rate [deg/s]');
title('Phase Portrait (Roll)');
grid on;
legend('Trajectory', 'Start', 'Target');

sgtitle('MPPI Attitude Control Diagnostics');

%% Print final state
fprintf('\n=== Final State ===\n');
fprintf('Preset: %s\n', dist_preset);
fprintf('Altitude: %.2f m (desired: %.2f m)\n', -X(3,end), -pos_des(3));
fprintf('Roll: %.2f deg\n', rad2deg(euler(1,end)));
fprintf('Pitch: %.2f deg\n', rad2deg(euler(2,end)));
fprintf('Yaw: %.2f deg\n', rad2deg(euler(3,end)));
fprintf('XY drift: %.2f m\n', norm(X(1:2,end)));
fprintf('===================\n');