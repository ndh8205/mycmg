function mag_meas = sensor_mag(R_b2n, bias, params)
% sensor_mag: Magnetometer measurement model
%
% Inputs:
%   R_b2n  - 3x3 rotation matrix (body to NED)
%   bias   - 3x1 magnetometer bias [Gauss]
%   params - parameter struct
%
% Output:
%   mag_meas - 3x1 measured magnetic field [Gauss] (body frame)
%
% Model:
%   mag_meas = R_n2b * mag_ned + bias + noise

%% Parameter extraction
mag_ned = params.env.mag_ned;                           % [Gauss]
noise_density = params.sensor.mag.noise_density;        % [Gauss/sqrt(Hz)]
rate = params.sensor.mag.rate;                          % [Hz]

%% True magnetic field in body frame
R_n2b = R_b2n';
mag_body = R_n2b * mag_ned;

%% Noise (discrete-time)
sigma = noise_density * sqrt(rate);
noise = sigma * randn(3,1);

%% Measurement
mag_meas = mag_body + bias + noise;

end