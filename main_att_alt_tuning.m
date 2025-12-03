%% main_tiltrotor_hover.m - Tiltrotor Hover Test
clear; clc; close all;
addpath(genpath('math'));
addpath(genpath('model'));
addpath(genpath('ctrl'));

%% Parameters
params = params_tiltrotor_init();

%% Simulation settings
sim_hz = 1000;
dt = 1/sim_hz;
t_end = 10;
t = 0:dt:t_end;
N = length(t);

%% Process noise covariance
Q = zeros(9,9);

%% Control gains
% Attitude PID
gains_att.Kp = [8; 8; 4];
gains_att.Ki = [0; 0; 0];
gains_att.Kd = [2; 2; 1];
gains_att.int_limit = [1; 1; 1];

% Altitude PID
gains_alt.Kp = 15;
gains_alt.Ki = 2;
gains_alt.Kd = 8;
gains_alt.int_limit = 5;

%% Control state initialization
ctrl_state.int_att = zeros(3,1);
ctrl_state.int_alt = 0;

%% Initial state (30x1)
m = params.drone.body.m;
g = params.env.g;
k_T = params.drone.motor.k_T;

% Hover motor speed (4 rotors, vertical thrust)
omega_hover = sqrt(m * g / (4 * k_T));

% Initial attitude: slight perturbation
euler0 = deg2rad([30; 20; 10]);
q0 = GetQUAT(euler0(3), euler0(2), euler0(1));
q0 = q0(:);

x0 = zeros(30, 1);
x0(1:3)   = [0; 0; -1];              % position (NED), 10m altitude
x0(4:6)   = [0; 0; 0];                % velocity (body)
x0(7:10)  = q0;                       % quaternion
x0(11:13) = [0; 0; 0];                % angular velocity
x0(14:17) = omega_hover * ones(4,1);  % motor speed
x0(18:26) = zeros(9, 1);              % biases
x0(27:28) = [0; 0];                   % tilt angles (hover = 0)
x0(29:30) = [0; 0];                   % tilt rates

%% Desired states
q_des = [1; 0; 0; 0];   % level attitude
alt_des = 10;           % 10m altitude

%% Data storage
X = zeros(30, N);
U = zeros(9, N);
X(:,1) = x0;

%% Simulation loop
for k = 1:N-1
    x_k = X(:,k);
    
    % Extract states
    pos_ned = x_k(1:3);
    vel_b = x_k(4:6);
    q = x_k(7:10);
    omega = x_k(11:13);
    tilt = x_k(27:28);
    
    % Current altitude (positive up)
    alt = -pos_ned(3);
    
    % Vertical velocity (NED to body, then z component)
    R_b2n = GetDCM_QUAT(q);
    vel_ned = R_b2n * vel_b;
    vel_z = -vel_ned(3);  % positive up
    
    % Attitude control
    [tau_cmd, ctrl_state] = attitude_pid(q_des, q, omega, ctrl_state, gains_att, dt);
    
    % Altitude control
    [thrust_cmd, ctrl_state] = altitude_pid_tilt(alt_des, alt, vel_z, ctrl_state, gains_alt, dt, params);
    
    % Control allocation (hover mode: tilt = 0)
    cmd_vec = [thrust_cmd; tau_cmd];
    omega_sq = tiltrotor_allocator(cmd_vec, tilt, params, 'inverse');
    omega_sq = max(omega_sq, 0);
    u_motor = sqrt(omega_sq);
    
    % Saturate motor commands
    omega_max = params.drone.motor.omega_b_max;
    omega_min = params.drone.motor.omega_b_min;
    u_motor = max(min(u_motor, omega_max), omega_min);
    
    % Tilt command (hover = 0)
    u_tilt = [0; 0];
    
    % Full input vector
    u = zeros(9, 1);
    u(1:4) = u_motor;
    u(5:6) = u_tilt;
    u(7:9) = 0;  % elevon, elevator
    
    U(:,k) = u;
    
    % Integration (use existing srk4)
    [x_next, ~, ~, ~, ~, ~, ~] = srk4(@tiltrotor_dynamics, x_k, u, Q, dt, params, k, dt);
    
    % Normalize quaternion and ensure continuity
    x_next(7:10) = x_next(7:10) / norm(x_next(7:10));
    x_next(7:10) = EnsQuatCont(x_next(7:10), x_k(7:10));
    
    X(:,k+1) = x_next;
end
U(:,end) = U(:,end-1);

%% Plot results
figure('Position', [100 100 1400 900]);

subplot(3,3,1);
plot(t, X(1:2,:));
xlabel('Time [s]'); ylabel('Position [m]');
legend('x','y'); title('Horizontal Position (NED)'); grid on;

subplot(3,3,2);
plot(t, -X(3,:), 'b', 'LineWidth', 1.5);
hold on;
yline(alt_des, 'r--', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('Altitude [m]');
title('Altitude'); grid on;
legend('Actual', 'Desired');

subplot(3,3,3);
euler = zeros(3,N);
for i = 1:N
    euler(:,i) = Quat2Euler(X(7:10,i));
end
plot(t, rad2deg(euler));
xlabel('Time [s]'); ylabel('Angle [deg]');
legend('\phi','\theta','\psi'); title('Euler Angles'); grid on;

subplot(3,3,4);
plot(t, rad2deg(X(11:13,:)));
xlabel('Time [s]'); ylabel('Angular Rate [deg/s]');
legend('p','q','r'); title('Angular Velocity'); grid on;

subplot(3,3,5);
plot(t, U(1:4,:));
xlabel('Time [s]'); ylabel('\omega [rad/s]');
legend('M0','M1','M2','M3'); title('Motor Commands'); grid on;

subplot(3,3,6);
plot(t, X(14:17,:));
xlabel('Time [s]'); ylabel('\omega [rad/s]');
legend('M0','M1','M2','M3'); title('Motor Speed (Actual)'); grid on;

subplot(3,3,7);
plot(t, rad2deg(X(27:28,:)));
xlabel('Time [s]'); ylabel('Tilt [deg]');
legend('Tilt 0', 'Tilt 2'); title('Tilt Angles'); grid on;

subplot(3,3,8);
vel_ned_hist = zeros(3,N);
for i = 1:N
    R = GetDCM_QUAT(X(7:10,i));
    vel_ned_hist(:,i) = R * X(4:6,i);
end
plot(t, vel_ned_hist);
xlabel('Time [s]'); ylabel('Velocity [m/s]');
legend('Vn','Ve','Vd'); title('Velocity (NED)'); grid on;

subplot(3,3,9);
% Position error
pos_err = sqrt(X(1,:).^2 + X(2,:).^2 + (X(3,:)+alt_des).^2);
plot(t, pos_err);
xlabel('Time [s]'); ylabel('Error [m]');
title('Position Error Magnitude'); grid on;

sgtitle('Tiltrotor Hover Test');

%% ========== LOCAL FUNCTIONS ==========

function [thrust_cmd, ctrl_state] = altitude_pid_tilt(alt_des, alt, vel_z, ctrl_state, gains, dt, params)
% Altitude PID for tiltrotor

    e_alt = alt_des - alt;
    e_dot = -vel_z;
    
    ctrl_state.int_alt = ctrl_state.int_alt + e_alt * dt;
    int_limit = gains.int_limit;
    ctrl_state.int_alt = max(min(ctrl_state.int_alt, int_limit), -int_limit);
    
    acc_cmd = gains.Kp * e_alt + gains.Ki * ctrl_state.int_alt + gains.Kd * e_dot;
    
    m = params.drone.body.m;
    g = params.env.g;
    thrust_cmd = m * (g + acc_cmd);
    
    % Thrust limits
    k_T = params.drone.motor.k_T;
    omega_max = params.drone.motor.omega_b_max;
    thrust_max = 4 * k_T * omega_max^2;
    thrust_cmd = max(min(thrust_cmd, thrust_max), 0);
end