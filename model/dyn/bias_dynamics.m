function b_dot = bias_dynamics(b, w, params)
% bias_dynamics: Sensor bias dynamics model
%
% Inputs:
%   b      - 9x1 bias states [b_gyro; b_accel; b_mag]
%   w      - 9x1 driving noise
%   params - parameter struct
%
% Output:
%   b_dot  - 9x1 bias derivatives
%
% Models:
%   Gyro/Accel: Random Walk
%     db/dt = w
%
%   Magnetometer: First-Order Gauss-Markov (FOGM)
%     db/dt = -1/tau * b + w

%% Extract biases
b_gyro  = b(1:3);
b_accel = b(4:6);
b_mag   = b(7:9);

%% Gyro bias - Random Walk
b_gyro_dot = w(1:3);

%% Accel bias - Random Walk
b_accel_dot = w(4:6);

%% Mag bias - FOGM
tau_mag = params.sensor.mag.bias_corr_time;  % [s]
b_mag_dot = -1/tau_mag * b_mag + w(7:9);

%% Output
b_dot = [b_gyro_dot; b_accel_dot; b_mag_dot];

end