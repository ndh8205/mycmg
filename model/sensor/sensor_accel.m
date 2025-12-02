function accel_meas = sensor_accel(F_body, bias, params)
% sensor_accel: Accelerometer measurement model
%
% Inputs:
%   F_body - 3x1 body force [N] (thrust direction: [0;0;-T])
%   bias   - 3x1 accel bias [m/s^2]
%   params - parameter struct
%
% Output:
%   accel_meas - 3x1 measured specific force [m/s^2] (body frame)
%
% Model:
%   Accelerometer measures specific force = F_applied / m
%   At hover: F_body = [0;0;-mg], so f = [0;0;-g]
%
%   accel_meas = f_specific + bias + noise

%% Parameter extraction
m = params.drone.body.m;
noise_density = params.sensor.imu.accel.noise_density;  % [m/s^2/sqrt(Hz)]
rate = params.sensor.imu.accel.rate;                    % [Hz]

%% Specific force = F_body / m
f_specific = F_body / m;

%% Noise (discrete-time)
sigma = noise_density * sqrt(rate);
noise = sigma * randn(3,1);

%% Measurement
accel_meas = f_specific + bias + noise;

end