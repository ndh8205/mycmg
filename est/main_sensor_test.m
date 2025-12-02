%% test_sensors.m - Sensor Model Validation (All Axes + Bias Dynamics)
clear; clc; close all;
addpath(genpath('math'));
addpath(genpath('model'));
addpath(genpath('ctrl'));

%% Parameters
params = params_init();

%% Simulation settings
sim_hz = 1000;
dt = 1/sim_hz;
t_end = 30;
t = 0:dt:t_end;
N = length(t);

%% Process noise covariance (continuous-time)
% Gyro bias random walk
sigma_bg = params.sensor.imu.gyro.random_walk;    % [rad/s^2/sqrt(Hz)]
% Accel bias random walk  
sigma_ba = params.sensor.imu.accel.random_walk;   % [m/s^3/sqrt(Hz)]
% Mag bias (FOGM driving noise)
sigma_bm = params.sensor.mag.random_walk;         % [Gauss/s/sqrt(Hz)]

Q = diag([sigma_bg^2 * ones(1,3), ...
          sigma_ba^2 * ones(1,3), ...
          sigma_bm^2 * ones(1,3)]);

%% Sensor rates
imu_rate = params.sensor.imu.gyro.rate;    % 200 Hz
gps_rate = params.sensor.gps.rate;          % 5 Hz
baro_rate = params.sensor.baro.rate;        % 50 Hz
mag_rate = params.sensor.mag.rate;          % 100 Hz

imu_skip = sim_hz / imu_rate;
gps_skip = sim_hz / gps_rate;
baro_skip = sim_hz / baro_rate;
mag_skip = sim_hz / mag_rate;

%% Initial state (26x1)
m = params.drone.body.m;
g = params.env.g;
k_T = params.drone.motor.k_T;

omega_hover = sqrt(m * g / (4 * k_T));

% Initial: hovering at 10m with slight attitude
euler0 = deg2rad([5; 3; 10]);
q0 = GetQUAT(euler0(3), euler0(2), euler0(1));
q0 = q0(:);

x0 = zeros(26, 1);
x0(1:3)   = [0; 0; -10];
x0(4:6)   = [0; 0; 0];
x0(7:10)  = q0;
x0(11:13) = [0.1; -0.05; 0.02];
x0(14:17) = omega_hover * ones(4,1);
x0(18:20) = [0.01; -0.005; 0.008];    % gyro bias init
x0(21:23) = [0.05; -0.03; 0.02];      % accel bias init
x0(24:26) = [0.001; -0.001; 0.002];   % mag bias init

%% Data storage
X = zeros(26, N);
X(:,1) = x0;

% True values
OMEGA_TRUE = zeros(3, N);
ACCEL_TRUE = zeros(3, N);
POS_TRUE = zeros(3, N);
VEL_TRUE = zeros(3, N);
MAG_TRUE = zeros(3, N);

% Biases
BIAS_GYRO = zeros(3, N);
BIAS_ACCEL = zeros(3, N);
BIAS_MAG = zeros(3, N);

% Sensor measurements
GYRO_MEAS = NaN(3, N);
ACCEL_MEAS = NaN(3, N);
GPS_POS_MEAS = NaN(3, N);
GPS_VEL_MEAS = NaN(3, N);
BARO_MEAS = NaN(1, N);
MAG_MEAS = NaN(3, N);

%% Control setup
gains_pos.Kp = [2; 2; 3];
gains_pos.Ki = [0.1; 0.1; 0.2];
gains_pos.Kd = [2; 2; 2];
gains_pos.int_limit = [2; 2; 2];

gains_att.Kp = [5; 5; 5];
gains_att.Ki = [0; 0; 0];
gains_att.Kd = [1; 1; 0.5];
gains_att.int_limit = [1; 1; 1];

ctrl_state.int_pos = zeros(3,1);
ctrl_state.int_att = zeros(3,1);
ctrl_state.dt = dt;

pos_des = [0; 0; -10];
yaw_des = deg2rad(10);

%% Simulation loop
for k = 1:N-1
    x_k = X(:,k);
    
    % Extract states
    pos_ned = x_k(1:3);
    vel_b = x_k(4:6);
    q = x_k(7:10);
    omega = x_k(11:13);
    omega_m = x_k(14:17);
    b_gyro = x_k(18:20);
    b_accel = x_k(21:23);
    b_mag = x_k(24:26);
    
    % DCM
    R_b2n = GetDCM_QUAT(q);
    vel_ned = R_b2n * vel_b;
    
    % Force calculation
    omega_sq = omega_m.^2;
    [cmd_vec, ~] = control_allocator(omega_sq, params, 'forward');
    T_total = cmd_vec(1);
    F_body = [0; 0; -T_total];
    
    % Store true values
    OMEGA_TRUE(:,k) = omega;
    f_specific = F_body/m - R_b2n'*[0;0;g];
    ACCEL_TRUE(:,k) = f_specific;
    POS_TRUE(:,k) = pos_ned;
    VEL_TRUE(:,k) = vel_ned;
    MAG_TRUE(:,k) = R_b2n' * params.env.mag_ned;
    
    % Store biases
    BIAS_GYRO(:,k) = b_gyro;
    BIAS_ACCEL(:,k) = b_accel;
    BIAS_MAG(:,k) = b_mag;
    
    % Sensor measurements at respective rates
    if mod(k-1, imu_skip) == 0
        GYRO_MEAS(:,k) = sensor_gyro(omega, b_gyro, params);
        ACCEL_MEAS(:,k) = sensor_accel(F_body, R_b2n, b_accel, params);
    end
    
    if mod(k-1, gps_skip) == 0
        [GPS_POS_MEAS(:,k), GPS_VEL_MEAS(:,k)] = sensor_gps(pos_ned, vel_ned, params);
    end
    
    if mod(k-1, baro_skip) == 0
        BARO_MEAS(k) = sensor_baro(pos_ned, params);
    end
    
    if mod(k-1, mag_skip) == 0
        MAG_MEAS(:,k) = sensor_mag(R_b2n, b_mag, params);
    end
    
    % Control
    [q_des, thrust_cmd, ctrl_state] = position_controller(...
        pos_des, pos_ned, vel_ned, yaw_des, ctrl_state, gains_pos, params);
    [tau_cmd, ctrl_state] = attitude_pid(q_des, q, omega, ctrl_state, gains_att, dt);
    
    cmd_vec = [thrust_cmd; tau_cmd];
    omega_sq = control_allocator(cmd_vec, params, 'inverse');
    omega_sq = max(omega_sq, 0);
    u = sqrt(omega_sq);
    
    omega_max = params.drone.motor.omega_b_max;
    omega_min = params.drone.motor.omega_b_min;
    u = max(min(u, omega_max), omega_min);
    
    % Integration
    [x_next, ~, ~, ~, ~, ~, ~] = srk4(@drone_dynamics, x_k, u, Q, dt, params, k, dt);
    x_next(7:10) = x_next(7:10) / norm(x_next(7:10));
    x_next(7:10) = EnsQuatCont(x_next(7:10), x_k(7:10));
    
    X(:,k+1) = x_next;
end

% Final step
OMEGA_TRUE(:,N) = X(11:13,N);
ACCEL_TRUE(:,N) = ACCEL_TRUE(:,N-1);
POS_TRUE(:,N) = X(1:3,N);
VEL_TRUE(:,N) = VEL_TRUE(:,N-1);
MAG_TRUE(:,N) = MAG_TRUE(:,N-1);
BIAS_GYRO(:,N) = X(18:20,N);
BIAS_ACCEL(:,N) = X(21:23,N);
BIAS_MAG(:,N) = X(24:26,N);

%% Plot 1: Gyroscope (3 axes + bias)
figure('Position', [50 50 1400 800], 'Name', 'Gyroscope');
labels = {'X', 'Y', 'Z'};
for i = 1:3
    subplot(3,2,2*i-1);
    plot(t, rad2deg(OMEGA_TRUE(i,:)), 'b', 'LineWidth', 1); hold on;
    plot(t, rad2deg(GYRO_MEAS(i,:)), 'r.', 'MarkerSize', 2);
    xlabel('Time [s]'); ylabel(['\omega_' labels{i} ' [deg/s]']);
    title(['Gyro ' labels{i} ' - Measurement']); grid on;
    legend('True', 'Measured');
    
    subplot(3,2,2*i);
    plot(t, rad2deg(BIAS_GYRO(i,:)), 'k', 'LineWidth', 1.5);
    xlabel('Time [s]'); ylabel(['b_{g' labels{i} '} [deg/s]']);
    title(['Gyro ' labels{i} ' - Bias']); grid on;
end
sgtitle('Gyroscope');

%% Plot 2: Accelerometer (3 axes + bias)
figure('Position', [100 50 1400 800], 'Name', 'Accelerometer');
for i = 1:3
    subplot(3,2,2*i-1);
    plot(t, ACCEL_TRUE(i,:), 'b', 'LineWidth', 1); hold on;
    plot(t, ACCEL_MEAS(i,:), 'r.', 'MarkerSize', 2);
    xlabel('Time [s]'); ylabel(['a_' labels{i} ' [m/s^2]']);
    title(['Accel ' labels{i} ' - Measurement']); grid on;
    legend('True', 'Measured');
    
    subplot(3,2,2*i);
    plot(t, BIAS_ACCEL(i,:), 'k', 'LineWidth', 1.5);
    xlabel('Time [s]'); ylabel(['b_{a' labels{i} '} [m/s^2]']);
    title(['Accel ' labels{i} ' - Bias']); grid on;
end
sgtitle('Accelerometer');

%% Plot 3: Magnetometer (3 axes + bias)
figure('Position', [150 50 1400 800], 'Name', 'Magnetometer');
for i = 1:3
    subplot(3,2,2*i-1);
    plot(t, MAG_TRUE(i,:), 'b', 'LineWidth', 1); hold on;
    plot(t, MAG_MEAS(i,:), 'm.', 'MarkerSize', 2);
    xlabel('Time [s]'); ylabel(['B_' labels{i} ' [Gauss]']);
    title(['Mag ' labels{i} ' - Measurement']); grid on;
    legend('True', 'Measured');
    
    subplot(3,2,2*i);
    plot(t, BIAS_MAG(i,:), 'k', 'LineWidth', 1.5);
    xlabel('Time [s]'); ylabel(['b_{m' labels{i} '} [Gauss]']);
    title(['Mag ' labels{i} ' - Bias (FOGM)']); grid on;
end
sgtitle('Magnetometer');

%% Plot 4: GPS Position (3 axes)
figure('Position', [200 50 1400 600], 'Name', 'GPS Position');
labels_pos = {'X (North)', 'Y (East)', 'Z (Down)'};
for i = 1:3
    subplot(1,3,i);
    plot(t, POS_TRUE(i,:), 'b', 'LineWidth', 1); hold on;
    plot(t, GPS_POS_MEAS(i,:), 'ro', 'MarkerSize', 3);
    xlabel('Time [s]'); ylabel(['Pos ' labels{i} ' [m]']);
    title(['GPS ' labels_pos{i}]); grid on;
    legend('True', 'Measured');
end
sgtitle('GPS Position');

%% Plot 5: GPS Velocity (3 axes)
figure('Position', [250 50 1400 600], 'Name', 'GPS Velocity');
for i = 1:3
    subplot(1,3,i);
    plot(t, VEL_TRUE(i,:), 'b', 'LineWidth', 1); hold on;
    plot(t, GPS_VEL_MEAS(i,:), 'ro', 'MarkerSize', 3);
    xlabel('Time [s]'); ylabel(['Vel ' labels{i} ' [m/s]']);
    title(['GPS Vel ' labels_pos{i}]); grid on;
    legend('True', 'Measured');
end
sgtitle('GPS Velocity');

%% Plot 6: Barometer
figure('Position', [300 50 600 400], 'Name', 'Barometer');
plot(t, -POS_TRUE(3,:), 'b', 'LineWidth', 1); hold on;
plot(t, BARO_MEAS, 'g.', 'MarkerSize', 3);
xlabel('Time [s]'); ylabel('Altitude [m]');
title('Barometer'); grid on;
legend('True', 'Measured');