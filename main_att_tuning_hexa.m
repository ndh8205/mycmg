%% main_hexa_test.m - Hexarotor Attitude & Altitude Control Test
clear; clc; close all;
addpath(genpath('math'));
addpath(genpath('model'));
addpath(genpath('ctrl'));
addpath(genpath('disturbance'));

%% Parameters (Hexarotor)
params = params_init('hexa');

%% Disturbance settings
dist_preset = 'level2';  % 'nominal', 'level1', 'level2', 'level3'
[params, dist_state] = dist_init(params, dist_preset);
if params.dist.uncertainty.enable
    params_true = apply_uncertainty(params);
else
    params_true = params;
end

%% Simulation settings
sim_hz = 1000;
dt = 1/sim_hz;
t_end = 10;
t = 0:dt:t_end;
N = length(t);

omega_bar2RPM = 60 / (2 * pi);

%% Process noise covariance
Q = zeros(9,9);

%% Control gains

% Position PID
gains_pos.Kp = [1; 1; 2];
gains_pos.Ki = [0.2; 0.2; 0.2];
gains_pos.Kd = [2; 2; 3];
gains_pos.int_limit = [2; 2; 2];

% Attitude PID
gains_att.Kp = [8; 8; 6];
gains_att.Ki = [0.1; 0.1; 0.1];
gains_att.Kd = [2; 2; 1.5];
gains_att.int_limit = [1; 1; 1];

% Altitude PID
gains_alt.Kp = gains_att.Kp(3);
gains_alt.Ki = gains_att.Ki(3);
gains_alt.Kd = gains_att.Kd(3);
gains_alt.int_limit = gains_att.int_limit(3);

%% Control state initialization
ctrl_state.int_att = zeros(3,1);
ctrl_state.int_alt = 0;

%% Initial state (28x1 for hexa)
m = params.drone.body.m;
g = params.env.g;
k_T = params.drone.motor.k_T;
n_motor = params.drone.body.n_motor;

omega_hover = sqrt(m * g / (n_motor * k_T));
fprintf('Hover motor speed: %.2f RPM\n', omega_hover * omega_bar2RPM);

% Initial euler: roll=20, pitch=15, yaw=10 deg
euler0 = deg2rad([20; 15; 10]);
q0 = GetQUAT(euler0(3), euler0(2), euler0(1));
q0 = q0(:);

x0 = zeros(28, 1);
x0(1:3)   = [0; 0; -10];              % position (NED), 10m altitude
x0(4:6)   = [0; 0; 0];                % velocity (body)
x0(7:10)  = q0;                       % quaternion
x0(11:13) = [0; 0; 0];                % angular velocity
x0(14:19) = omega_hover * ones(6,1);  % motor speed (6 motors)
x0(20:28) = zeros(9, 1);              % biases

%% Desired states
q_des = [1; 0; 0; 0];   % level attitude
alt_des = 10;           % maintain 10m altitude

%% Data storage
X = zeros(28, N);
U = zeros(6, N);
TAU_DIST = zeros(3, N);
F_DIST = zeros(3, N);
X(:,1) = x0;

%% Simulation loop
fprintf('Running simulation (dist: %s)...\n', dist_preset);
for k = 1:N-1
    x_k = X(:,k);
    
    % Extract states
    pos_ned = x_k(1:3);
    vel_b = x_k(4:6);
    q = x_k(7:10);
    omega = x_k(11:13);
    
    % Current altitude (positive up)
    alt = -pos_ned(3);
    
    % Vertical velocity
    R_b2n = GetDCM_QUAT(q);
    vel_ned = R_b2n * vel_b;
    vel_z = -vel_ned(3);
    
    % Attitude control
    [tau_cmd, ctrl_state] = attitude_pid(q_des, q, omega, ctrl_state, gains_att, dt);
    
    % Altitude control
    [thrust_cmd, ctrl_state] = altitude_pid(alt_des, alt, vel_z, ctrl_state, gains_alt, dt, params);
    
    % Control allocation
    cmd_vec = [thrust_cmd; tau_cmd];
    omega_sq = control_allocator(cmd_vec, params, 'inverse');
    omega_sq = max(omega_sq, 0);
    u = sqrt(omega_sq);
    
    % Saturate motor commands
    omega_max = params.drone.motor.omega_b_max;
    omega_min = params.drone.motor.omega_b_min;
    u = max(min(u, omega_max), omega_min);
    U(:,k) = u;
    
    % Log disturbance
    if params.dist.enable
        [tau_d, ~] = dist_torque(t(k), params, dist_state);
        [F_d, ~] = dist_wind(vel_b, R_b2n, t(k), dt, params, dist_state);
        TAU_DIST(:,k) = tau_d;
        F_DIST(:,k) = F_d;
    end
    
    % Integration (use params_true for dynamics)
    [x_next, ~, ~, ~, ~, ~, dist_state] = srk4(@drone_dynamics, x_k, u, Q, dt, params_true, k, dt, t(k), dist_state);
    
    % Normalize quaternion and ensure continuity
    x_next(7:10) = x_next(7:10) / norm(x_next(7:10));
    x_next(7:10) = EnsQuatCont(x_next(7:10), x_k(7:10));
    
    X(:,k+1) = x_next;
end
U(:,end) = U(:,end-1);
fprintf('Simulation complete.\n');

%% Plot results
figure('Position', [100 100 1200 800]);

subplot(3,3,1);
plot(t, -X(3,:), 'b', 'LineWidth', 1.5);
hold on;
yline(alt_des, 'r--', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('Altitude [m]');
title('Altitude'); grid on;
legend('Actual', 'Desired');

subplot(3,3,2);
euler = zeros(3,N);
for k = 1:N
    euler(:,k) = Quat2Euler(X(7:10,k));
end
plot(t, rad2deg(euler));
xlabel('Time [s]'); ylabel('Angle [deg]');
legend('\phi','\theta','\psi'); 
title('Euler Angles'); grid on;
yline(0, 'k--');

subplot(3,3,3);
plot(t, rad2deg(X(11:13,:)));
xlabel('Time [s]'); ylabel('Angular Rate [deg/s]');
legend('p','q','r'); 
title('Angular Velocity (Body)'); grid on;

subplot(3,3,4);
plot(t, U * omega_bar2RPM);
xlabel('Time [s]'); ylabel('Motor Speed [RPM]');
legend('M1','M2','M3','M4','M5','M6');
title('Motor Commands'); grid on;

subplot(3,3,5);
plot(t, X(14:19,:) * omega_bar2RPM);
xlabel('Time [s]'); ylabel('Motor Speed [RPM]');
legend('M1','M2','M3','M4','M5','M6');
title('Motor Speed (Actual)'); grid on;

subplot(3,3,6);
plot(t, X(1:2,:));
xlabel('Time [s]'); ylabel('Position [m]');
legend('x','y');
title('Horizontal Position (NED)'); grid on;

subplot(3,3,7);
plot(t, TAU_DIST');
xlabel('Time [s]'); ylabel('Torque [Nm]');
legend('\tau_x','\tau_y','\tau_z');
title('Torque Disturbance'); grid on;

subplot(3,3,8);
plot(t, F_DIST');
xlabel('Time [s]'); ylabel('Force [N]');
legend('F_x','F_y','F_z');
title('Wind Force (Body)'); grid on;

subplot(3,3,9);
plot(t, vecnorm(X(1:2,:)), 'b', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('|XY Error| [m]');
title('Horizontal Drift'); grid on;

sgtitle(sprintf('Hexarotor Attitude & Altitude Control (Dist: %s)', dist_preset));

%% Print final state
fprintf('\n=== Final State ===\n');
fprintf('Preset: %s\n', dist_preset);
fprintf('Altitude: %.2f m (desired: %.2f m)\n', -X(3,end), alt_des);
fprintf('Roll: %.2f deg\n', rad2deg(euler(1,end)));
fprintf('Pitch: %.2f deg\n', rad2deg(euler(2,end)));
fprintf('Yaw: %.2f deg\n', rad2deg(euler(3,end)));
fprintf('XY drift: %.2f m\n', norm(X(1:2,end)));