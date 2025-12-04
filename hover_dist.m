%% sim_hexa_hover.m
% Hexacopter hover simulation with disturbance
% Tests drone_dynamics + srk4 + disturbance system

clear; clc; close all;

addpath(genpath('math'));
addpath(genpath('model'));
addpath(genpath('ctrl'));
addpath(genpath('disturbance'));

%% 1. Simulation parameters
dt = 0.01;          % Time step [s]
t_end = 30;         % Simulation duration [s]
t_vec = 0:dt:t_end;
N = length(t_vec);

%% 2. Initialize system
% Nominal parameters (for controller)
params_nom = params_init('hexa');

% Disturbance preset: 'nominal', 'level1', 'level2', 'level3'
dist_preset = 'level3';
[params_nom, dist_state] = dist_init(params_nom, dist_preset);

% True parameters (for dynamics) - with uncertainty
if params_nom.dist.uncertainty.enable
    params_true = apply_uncertainty(params_nom);
else
    params_true = params_nom;
end

%% 3. Initial state (hexarotor: 28 states)
% Hover at -10m altitude (NED: z = -10)
pos_init = [0; 0; -10];         % [m] NED
vel_init = [0; 0; 0];           % [m/s] body
quat_init = [1; 0; 0; 0];       % Level attitude
omega_init = [0; 0; 0];         % [rad/s]

% Hover motor speed
m = params_nom.drone.body.m;
g = params_nom.env.g;
k_T = params_nom.drone.motor.k_T;
n_motor = 6;
T_hover = m * g;
omega_m_hover = sqrt(T_hover / (n_motor * k_T));
omega_m_init = omega_m_hover * ones(6,1);

% Bias states
bias_init = zeros(9,1);

% Full state
x0 = [pos_init; vel_init; quat_init; omega_init; omega_m_init; bias_init];

%% 4. Process noise covariance (for bias random walk)
Q = diag([
    params_nom.sensor.imu.gyro.random_walk^2 * ones(1,3), ...
    params_nom.sensor.imu.accel.random_walk^2 * ones(1,3), ...
    params_nom.sensor.mag.random_walk^2 * ones(1,3)
]);

%% 5. Controller gains (simple hover hold)
% Attitude PID
att_gains.Kp = [5; 5; 3];
att_gains.Ki = [0.5; 0.5; 0.2];
att_gains.Kd = [2; 2; 1];
att_gains.int_limit = [0.5; 0.5; 0.3];

% Position PID
pos_gains.Kp = [2; 2; 4];
pos_gains.Ki = [0.1; 0.1; 0.2];
pos_gains.Kd = [1.5; 1.5; 2];
pos_gains.int_limit = [2; 2; 3];

% Controller state
ctrl_state.int_att = zeros(3,1);
ctrl_state.int_pos = zeros(3,1);
ctrl_state.dt = dt;

%% 6. Reference
pos_des = [0; 0; -10];  % Hover at 10m
yaw_des = 0;

%% 7. Preallocate logs
x_log = zeros(length(x0), N);
u_log = zeros(6, N);
tau_dist_log = zeros(3, N);
F_dist_log = zeros(3, N);

%% 8. Simulation loop
x = x0;
fprintf('Running simulation: %s preset...\n', dist_preset);

for k = 1:N
    t = t_vec(k);
    
    % Log state
    x_log(:,k) = x;
    
    % Extract states for controller
    pos = x(1:3);
    vel_b = x(4:6);
    quat = x(7:10);
    omega = x(11:13);
    R_b2n = GetDCM_QUAT(quat);
    vel_ned = R_b2n * vel_b;
    
    % Position controller -> desired attitude & thrust
    [q_des, thrust_cmd, ctrl_state] = position_pid(pos_des, pos, vel_ned, yaw_des, ctrl_state, pos_gains, params_nom);
    
    % Attitude controller -> torque command
    [tau_cmd, ctrl_state] = attitude_pid(q_des, quat, omega, ctrl_state, att_gains, dt);
    
    % Control allocation
    cmd_vec = [thrust_cmd; tau_cmd];
    [omega_sq_cmd, ~] = control_allocator(cmd_vec, params_nom, 'inverse');
    omega_sq_cmd = max(omega_sq_cmd, 0);  % No negative thrust
    u_cmd = sqrt(omega_sq_cmd);
    
    % Saturate motor commands
    u_cmd = min(max(u_cmd, params_nom.drone.motor.omega_b_min), params_nom.drone.motor.omega_b_max);
    u_log(:,k) = u_cmd;
    
    % Log disturbance (before dynamics call)
    if params_nom.dist.enable
        [tau_d, ~] = dist_torque(t, params_nom, dist_state);
        [F_d, ~] = dist_wind(vel_b, R_b2n, t, dt, params_nom, dist_state);
        tau_dist_log(:,k) = tau_d;
        F_dist_log(:,k) = F_d;
    end
    
    % Integrate dynamics (use true params for plant)
    [x, ~, ~, ~, ~, ~, dist_state] = srk4(@drone_dynamics, x, u_cmd, Q, dt, params_true, k, dt, t, dist_state);
    
    % Normalize quaternion
    x(7:10) = x(7:10) / norm(x(7:10));
end

fprintf('Simulation complete.\n');

%% 9. Extract results
pos_log = x_log(1:3, :);
vel_log = x_log(4:6, :);
quat_log = x_log(7:10, :);
omega_log = x_log(11:13, :);
omega_m_log = x_log(14:19, :);

% Convert quaternion to Euler
euler_log = zeros(3, N);
for k = 1:N
    euler_log(:,k) = Quat2Euler(quat_log(:,k));
end
euler_log = rad2deg(euler_log);

%% 10. Plot results
figure('Name', 'Hexacopter Hover Simulation', 'Position', [50 50 1400 800]);

% Position
subplot(3,3,1);
plot(t_vec, pos_log(1,:), 'r', t_vec, pos_log(2,:), 'g', t_vec, pos_log(3,:), 'b', 'LineWidth', 1);
xlabel('Time [s]'); ylabel('Position [m]');
title('Position (NED)');
legend('X', 'Y', 'Z'); grid on;

% Altitude detail
subplot(3,3,2);
plot(t_vec, -pos_log(3,:), 'b', 'LineWidth', 1);
hold on; yline(-pos_des(3), 'r--', 'LineWidth', 1);
xlabel('Time [s]'); ylabel('Altitude [m]');
title('Altitude');
legend('Actual', 'Desired'); grid on;

% Euler angles
subplot(3,3,3);
plot(t_vec, euler_log(1,:), 'r', t_vec, euler_log(2,:), 'g', t_vec, euler_log(3,:), 'b', 'LineWidth', 1);
xlabel('Time [s]'); ylabel('Angle [deg]');
title('Euler Angles');
legend('\phi', '\theta', '\psi'); grid on;

% Angular velocity
subplot(3,3,4);
plot(t_vec, omega_log(1,:), 'r', t_vec, omega_log(2,:), 'g', t_vec, omega_log(3,:), 'b', 'LineWidth', 1);
xlabel('Time [s]'); ylabel('\omega [rad/s]');
title('Angular Velocity');
legend('p', 'q', 'r'); grid on;

% Motor speeds
subplot(3,3,5);
plot(t_vec, omega_m_log', 'LineWidth', 1);
xlabel('Time [s]'); ylabel('\omega_m [rad/s]');
title('Motor Speeds');
legend('M1','M2','M3','M4','M5','M6'); grid on;

% XY trajectory
subplot(3,3,6);
plot(pos_log(1,:), pos_log(2,:), 'b', 'LineWidth', 1);
hold on; plot(0, 0, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
xlabel('X [m]'); ylabel('Y [m]');
title('XY Trajectory');
axis equal; grid on;

% Torque disturbance
subplot(3,3,7);
plot(t_vec, tau_dist_log', 'LineWidth', 1);
xlabel('Time [s]'); ylabel('Torque [Nm]');
title('Torque Disturbance');
legend('\tau_x', '\tau_y', '\tau_z'); grid on;

% Wind force
subplot(3,3,8);
plot(t_vec, F_dist_log', 'LineWidth', 1);
xlabel('Time [s]'); ylabel('Force [N]');
title('Wind Force (Body)');
legend('F_x', 'F_y', 'F_z'); grid on;

% Position error
subplot(3,3,9);
pos_err = pos_log - pos_des;
plot(t_vec, vecnorm(pos_err), 'k', 'LineWidth', 1);
xlabel('Time [s]'); ylabel('|Error| [m]');
title('Position Error Magnitude');
grid on;

sgtitle(sprintf('Hexacopter Hover - Disturbance: %s', dist_preset));

%% 11. Print summary
fprintf('\n=== Simulation Summary ===\n');
fprintf('Preset: %s\n', dist_preset);
fprintf('Duration: %.1f s\n', t_end);
fprintf('Final position error: %.4f m\n', norm(pos_log(:,end) - pos_des));
fprintf('Max position error: %.4f m\n', max(vecnorm(pos_err)));
fprintf('Max attitude error: %.2f deg\n', max(abs(euler_log(:))));
fprintf('==========================\n');