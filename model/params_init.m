function params = params_init(drone_type)
% params_init: 드론, 환경, 센서 파라미터 초기화
%
% Input:
%   drone_type - 'quad' (default) or 'hexa'
%
% Output:
%   params.drone  - 드론 물성치
%   params.env    - 환경 파라미터
%   params.sensor - 센서 파라미터

if nargin < 1
    drone_type = 'quad';
end

%% ========== DRONE ==========
params.drone.type = drone_type;

if strcmp(drone_type, 'hexa')
    %% Hexarotor (논문 파라미터)
    % Body
    params.drone.body.m = 6.2;  % [kg]
    params.drone.body.J = diag([0.17, 0.175, 0.263]);  % [kg*m^2]
    
    % Geometry
    params.drone.body.L = 0.96;  % [m] arm length
    params.drone.body.n_motor = 6;
    
    % Motor
    params.drone.motor.k_T = 0.03;          % Thrust coeff [N/(rad/s)^2]
    params.drone.motor.k_M = 0.001/0.03;    % Moment coeff ratio: b/k
    params.drone.motor.tau_up = 0.0125;     % Time constant (up) [s]
    params.drone.motor.tau_down = 0.025;    % Time constant (down) [s]
    params.drone.motor.omega_b_max = 800;   % Max angular velocity [rad/s]
    params.drone.motor.omega_b_min = 0;     % Min angular velocity [rad/s]
    
    % Aerodynamic drag
    params.drone.aero.kd = 0.25;  % [N/(m/s)]
    
else
    %% Quadrotor (기존)
    % Body
    params.drone.body.m = 1.5;  % [kg]
    params.drone.body.J = diag([0.029125, 0.029125, 0.055225]);  % [kg*m^2]
    
    % Geometry
    params.drone.body.n_motor = 4;
    
    % Motor
    params.drone.motor.k_T = 5.84e-06;      % Thrust coeff: T = k_T * omega^2 [N/(rad/s)^2]
    params.drone.motor.k_M = 0.06;          % Moment coeff: tau = k_M * T [m]
    params.drone.motor.tau_up = 0.0125;     % Time constant (up) [s]
    params.drone.motor.tau_down = 0.025;    % Time constant (down) [s]
    params.drone.motor.omega_b_max = 1100;  % Max angular velocity [rad/s]
    params.drone.motor.omega_b_min = 0;     % Min angular velocity [rad/s]
end

%% ========== ENVIRONMENT ==========
params.env.g = 9.81;        % Gravity [m/s^2]
params.env.rho = 1.225;     % Air density [kg/m^3]

% Magnetic field (NED frame) - Seoul approx.
params.env.mag_ned = [0.22; -0.04; 0.43];  % [Gauss] normalized

%% ========== SENSOR ==========
% IMU - Gyroscope
params.sensor.imu.gyro.noise_density = 0.0003394;   % [rad/s/sqrt(Hz)] ARW
params.sensor.imu.gyro.random_walk = 3.8785e-05;    % [rad/s^2/sqrt(Hz)]
params.sensor.imu.gyro.bias_init = [0; 0; 0];       % [rad/s]
params.sensor.imu.gyro.rate = 200;                  % [Hz]

% IMU - Accelerometer
params.sensor.imu.accel.noise_density = 0.004;      % [m/s^2/sqrt(Hz)] VRW
params.sensor.imu.accel.random_walk = 6.0e-03;      % [m/s^3/sqrt(Hz)]
params.sensor.imu.accel.bias_init = [0; 0; 0];      % [m/s^2]
params.sensor.imu.accel.rate = 200;                 % [Hz]

% GPS
params.sensor.gps.pos_std = [0.3; 0.3; 0.5];        % [m] position noise
params.sensor.gps.vel_std = [0.05; 0.05; 0.1];      % [m/s] velocity noise
params.sensor.gps.rate = 5;                         % [Hz]

% Barometer
params.sensor.baro.noise_std = 0.5;     % [m] altitude noise
params.sensor.baro.drift = 0;           % [Pa/s]
params.sensor.baro.rate = 50;           % [Hz]

% Magnetometer
params.sensor.mag.noise_density = 0.0004;       % [Gauss/sqrt(Hz)]
params.sensor.mag.random_walk = 6.4e-06;        % [Gauss/s/sqrt(Hz)]
params.sensor.mag.bias_corr_time = 600;         % [s]
params.sensor.mag.rate = 100;                   % [Hz]

end