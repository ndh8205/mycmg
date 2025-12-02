function [pos_meas, vel_meas] = sensor_gps(pos_ned, vel_ned, params)
% sensor_gps: GPS measurement model
%
% Inputs:
%   pos_ned - 3x1 true position [m] (NED frame)
%   vel_ned - 3x1 true velocity [m/s] (NED frame)
%   params  - parameter struct
%
% Outputs:
%   pos_meas - 3x1 measured position [m]
%   vel_meas - 3x1 measured velocity [m/s]
%
% Model:
%   pos_meas = pos_ned + noise_pos
%   vel_meas = vel_ned + noise_vel

%% Parameter extraction
pos_std = params.sensor.gps.pos_std;  % [m]
vel_std = params.sensor.gps.vel_std;  % [m/s]

%% Noise
noise_pos = pos_std .* randn(3,1);
noise_vel = vel_std .* randn(3,1);

%% Measurement
pos_meas = pos_ned + noise_pos;
vel_meas = vel_ned + noise_vel;

end