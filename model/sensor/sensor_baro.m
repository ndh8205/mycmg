function alt_meas = sensor_baro(pos_ned, params)
% sensor_baro: Barometer measurement model
%
% Inputs:
%   pos_ned - 3x1 true position [m] (NED frame)
%   params  - parameter struct
%
% Output:
%   alt_meas - measured altitude [m] (positive up)
%
% Model:
%   alt_meas = -pos_ned(3) + noise

%% Parameter extraction
noise_std = params.sensor.baro.noise_std;  % [m]

%% Noise
noise = noise_std * randn();

%% Measurement (altitude = -z in NED)
alt_meas = -pos_ned(3) + noise;

end