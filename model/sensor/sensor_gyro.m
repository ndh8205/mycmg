function omega_meas = sensor_gyro(omega_true, bias, params)
% sensor_gyro: Gyroscope measurement model
%
% Inputs:
%   omega_true - 3x1 true angular velocity [rad/s] (body frame)
%   bias       - 3x1 gyro bias [rad/s]
%   params     - parameter struct
%
% Output:
%   omega_meas - 3x1 measured angular velocity [rad/s]
%
% Model:
%   omega_meas = omega_true + bias + noise

%% Parameter extraction
noise_density = params.sensor.imu.gyro.noise_density;  % [rad/s/sqrt(Hz)]
rate = params.sensor.imu.gyro.rate;                    % [Hz]

%% Noise (discrete-time)
sigma = noise_density * sqrt(rate);
noise = sigma * randn(3,1);

%% Measurement
omega_meas = omega_true + bias + noise;

end