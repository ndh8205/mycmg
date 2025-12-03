function params = params_tiltrotor_init()
% params_tiltrotor_init: 틸트로터 파라미터 초기화
%
% Output:
%   params.drone  - 기체 물성치
%   params.env    - 환경 파라미터
%   params.aero   - 공력 파라미터
%   params.sensor - 센서 파라미터

%% ========== DRONE ==========
% Body (from SDF)
params.drone.body.m = 5;  % [kg]
params.drone.body.J = diag([0.197563, 0.1458929, 0.1477]);  % [kg*m^2]

% Rotor configuration (consistent with tiltrotor_allocator.m)
%   rotor 0: front-right (tiltable, CCW)
%   rotor 1: back-left   (fixed, CCW)
%   rotor 2: front-left  (tiltable, CW)
%   rotor 3: back-right  (fixed, CW)
%
% Position [x, y, z] in body frame (NED: X-fwd, Y-right, Z-down)
params.drone.rotor.pos = [
     0.35,  0.35, -0.07;   % rotor 0: front-right
    -0.35, -0.35, -0.07;   % rotor 1: back-left
     0.35, -0.35, -0.07;   % rotor 2: front-left
    -0.35,  0.35, -0.07;   % rotor 3: back-right
];
params.drone.rotor.dir = [1; 1; -1; -1];      % CCW=+1, CW=-1 (yaw torque sign)
params.drone.rotor.tiltable = [1; 0; 1; 0];   % 1=tiltable, 0=fixed

% Motor parameters (from SDF)
params.drone.motor.k_T = 2e-05;       % Thrust coeff [N/(rad/s)^2]
params.drone.motor.k_M = 0.06;        % Moment coeff [m]
params.drone.motor.tau_up = 0.0125;   % Time constant (spin up) [s]
params.drone.motor.tau_down = 0.025;  % Time constant (spin down) [s]
params.drone.motor.omega_b_max = 1500;  % Max angular velocity [rad/s]
params.drone.motor.omega_b_min = 0;     % Min angular velocity [rad/s]

% Tilt actuator parameters
params.drone.tilt.tau = 0.1;          % Servo time constant [s]
params.drone.tilt.max = 1.5;          % Max tilt angle [rad] (~86 deg, forward)
params.drone.tilt.min = -0.1;         % Min tilt angle [rad] (slight back-tilt)
params.drone.tilt.rate_max = 1.0;     % Max tilt rate [rad/s]

%% ========== ENVIRONMENT ==========
params.env.g = 9.81;        % Gravity [m/s^2]
params.env.rho = 1.2041;    % Air density [kg/m^3] (from SDF)

% Magnetic field (NED frame) - Seoul approx.
params.env.mag_ned = [0.22; -0.04; 0.43];  % [Gauss]

%% ========== AERODYNAMICS ==========
% Main wing (left and right, symmetric)
params.aero.wing.area = 0.5;              % [m^2] per side
params.aero.wing.a0 = 0.05984281113;      % Zero-lift AoA [rad]
params.aero.wing.CL_alpha = 4.752798721;  % Lift curve slope [1/rad]
params.aero.wing.CD_alpha = 0.6417112299; % Drag coefficient slope
params.aero.wing.CD_0 = 0.02;             % Zero-lift drag
params.aero.wing.alpha_stall = 0.3391428111;  % Stall angle [rad] (~19.4 deg)
params.aero.wing.CL_stall = -3.85;        % Post-stall CL slope
params.aero.wing.CD_stall = -0.9233984055;% Post-stall CD slope

% Wing center of pressure (body frame, NED: Y+ = right)
params.aero.wing.cp_left  = [-0.05; -0.3; -0.05];  % left wing (Y < 0)
params.aero.wing.cp_right = [-0.05;  0.3; -0.05];  % right wing (Y > 0)

% Elevon (left/right)
params.aero.elevon.CL_delta = 1.0;    % Control effectiveness [1/rad]
params.aero.elevon.max = 0.53;        % Max deflection [rad] (~30 deg)
params.aero.elevon.min = -0.53;       % Min deflection [rad]

% Elevator
params.aero.elevator.area = 0.01;     % [m^2]
params.aero.elevator.cp = [-0.5; 0; 0];  % Center of pressure
params.aero.elevator.CL_delta = 12.0; % Control effectiveness [1/rad]
params.aero.elevator.max = 0.53;      % Max deflection [rad]
params.aero.elevator.min = -0.53;     % Min deflection [rad]

% Rudder (minimal effect in this config)
params.aero.rudder.area = 0.02;
params.aero.rudder.cp = [-0.5; 0; -0.05];

%% ========== FLIGHT MODE ==========
% Transition speed thresholds
params.flight.V_transition_start = 8;   % [m/s] Start transition
params.flight.V_transition_end = 15;    % [m/s] Complete transition
params.flight.V_stall = 12;             % [m/s] Stall speed

%% ========== SENSOR ==========
% IMU - Gyroscope
params.sensor.imu.gyro.noise_density = 0.0003394;
params.sensor.imu.gyro.random_walk = 3.8785e-05;
params.sensor.imu.gyro.bias_init = [0; 0; 0];
params.sensor.imu.gyro.rate = 200;

% IMU - Accelerometer
params.sensor.imu.accel.noise_density = 0.004;
params.sensor.imu.accel.random_walk = 6.0e-03;
params.sensor.imu.accel.bias_init = [0; 0; 0];
params.sensor.imu.accel.rate = 200;

% GPS
params.sensor.gps.pos_std = [0.3; 0.3; 0.5];
params.sensor.gps.vel_std = [0.05; 0.05; 0.1];
params.sensor.gps.rate = 5;

% Barometer
params.sensor.baro.noise_std = 0.5;
params.sensor.baro.drift = 0;
params.sensor.baro.rate = 50;

% Magnetometer
params.sensor.mag.noise_density = 0.0004;
params.sensor.mag.random_walk = 6.4e-06;
params.sensor.mag.bias_corr_time = 600;
params.sensor.mag.rate = 100;

% Airspeed
params.sensor.airspeed.noise_std = 0.5;  % [m/s]
params.sensor.airspeed.rate = 50;        % [Hz]

end