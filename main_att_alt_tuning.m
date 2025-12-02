%% main.m - 6-DoF Drone Simulation with Attitude & Altitude Control
clear; clc; close all;
addpath(genpath('math'));
addpath(genpath('model'));
addpath(genpath('ctrl'));

%% Parameters
params = params_init();

%% Simulation settings
sim_hz = 1000;
dt = 1/sim_hz;
t_end = 3;
t = 0:dt:t_end;
N = length(t);

%% Process noise covariance
Q = zeros(9,9);

%% Control gains
% Attitude PID
gains_att.Kp = [5; 5; 5];
gains_att.Ki = [0; 0; 0];
gains_att.Kd = [1; 1; 0.5];
gains_att.int_limit = [1; 1; 1];

% Altitude PID
gains_alt.Kp = 15;
gains_alt.Ki = 0;
gains_alt.Kd = 7;
gains_alt.int_limit = 5;

%% Control state initialization
ctrl_state.int_att = zeros(3,1);
ctrl_state.int_alt = 0;

%% Initial state (26x1)
m = params.drone.body.m;
g = params.env.g;
k_T = params.drone.motor.k_T;

omega_hover = sqrt(m * g / (4 * k_T));

% Initial euler: roll=30, pitch=20, yaw=10 deg
euler0 = deg2rad([30; 20; 10]);
q0 = GetQUAT(euler0(3), euler0(2), euler0(1));  % yaw, pitch, roll
q0 = q0(:);

x0 = zeros(26, 1);
x0(1:3)   = [0; 0; -1];              % position (NED), 10m altitude
x0(4:6)   = [0; 0; 0];                % velocity (body)
x0(7:10)  = q0;                       % quaternion
x0(11:13) = [0; 0; 0];                % angular velocity
x0(14:17) = omega_hover * ones(4,1);  % motor speed
x0(18:26) = zeros(9, 1);              % biases

%% Desired states
q_des = [1; 0; 0; 0];   % level attitude
alt_des = 10;            % maintain 10m altitude

%% Data storage
X = zeros(26, N);
U = zeros(4, N);
X(:,1) = x0;

%% Simulation loop
for k = 1:N-1
    x_k = X(:,k);
    
    % Extract states
    pos_ned = x_k(1:3);
    vel_b = x_k(4:6);
    q = x_k(7:10);
    omega = x_k(11:13);
    
    % Current altitude (positive up)
    alt = -pos_ned(3);
    
    % Vertical velocity (NED to body, then z component)
    R_b2n = GetDCM_QUAT(q);
    vel_ned = R_b2n * vel_b;
    vel_z = -vel_ned(3);  % positive up
    
    % Attitude control
    [tau_cmd, ctrl_state] = attitude_pid(q_des, q, omega, ctrl_state, gains_att, dt);
    
    % Altitude control
    [thrust_cmd, ctrl_state] = altitude_pid(alt_des, alt, vel_z, ctrl_state, gains_alt, dt, params);
    
    % Control allocation
    cmd_vec = [thrust_cmd; tau_cmd];
    omega_sq = control_allocator(cmd_vec, params, 'inverse');
    omega_sq = max(omega_sq, 0);  % ensure non-negative
    u = sqrt(omega_sq);
    
    % Saturate motor commands
    omega_max = params.drone.motor.omega_b_max;
    omega_min = params.drone.motor.omega_b_min;
    u = max(min(u, omega_max), omega_min);
    U(:,k) = u;
    
    % Integration
    [x_next, ~, ~, ~, ~, ~, ~] = srk4(@drone_dynamics, x_k, u, Q, dt, params, k, dt);
    
    % Normalize quaternion and ensure continuity
    x_next(7:10) = x_next(7:10) / norm(x_next(7:10));
    x_next(7:10) = EnsQuatCont(x_next(7:10), x_k(7:10));
    
    X(:,k+1) = x_next;
end
U(:,end) = U(:,end-1);

%% Plot results
figure('Position', [100 100 1200 800]);

subplot(3,2,1);
plot(t, X(1:3,:));
xlabel('Time [s]'); ylabel('Position [m]');
legend('x','y','z'); title('Position (NED)'); grid on;

subplot(3,2,2);
plot(t, -X(3,:));
xlabel('Time [s]'); ylabel('Altitude [m]');
title('Altitude'); grid on;
yline(alt_des, 'r--', 'Desired');

subplot(3,2,3);
euler = zeros(3,N);
for k = 1:N
    euler(:,k) = Quat2Euler(X(7:10,k));
end
plot(t, rad2deg(euler));
xlabel('Time [s]'); ylabel('Angle [deg]');
legend('\phi','\theta','\psi'); title('Euler Angles'); grid on;
yline(0, 'r--');

subplot(3,2,4);
plot(t, rad2deg(X(11:13,:)));
xlabel('Time [s]'); ylabel('Angular Rate [deg/s]');
legend('p','q','r'); title('Angular Velocity (Body)'); grid on;

subplot(3,2,5);
plot(t, U);
xlabel('Time [s]'); ylabel('\omega_m [rad/s]');
legend('M1','M2','M3','M4'); title('Motor Commands'); grid on;

subplot(3,2,6);
plot(t, X(14:17,:));
xlabel('Time [s]'); ylabel('\omega_m [rad/s]');
legend('M1','M2','M3','M4'); title('Motor Speed (Actual)'); grid on;