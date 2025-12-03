function tilt_ddot = tilt_actuator_dynamics(tilt, tilt_dot, u_tilt, params)
% tilt_actuator_dynamics: Tilt servo dynamics with rate limiting
%
% Inputs:
%   tilt     - 2x1 current tilt angles [rad]
%   tilt_dot - 2x1 current tilt rates [rad/s]
%   u_tilt   - 2x1 commanded tilt angles [rad]
%   params   - parameter struct
%
% Output:
%   tilt_ddot - 2x1 tilt accelerations [rad/s^2]

tau = params.drone.tilt.tau;
tilt_max = params.drone.tilt.max;
tilt_min = params.drone.tilt.min;
rate_max = params.drone.tilt.rate_max;

% Saturate command
u_sat = max(min(u_tilt, tilt_max), tilt_min);

% Desired rate (first-order response)
tilt_dot_des = (u_sat - tilt) / tau;

% Rate limiting
tilt_dot_des = max(min(tilt_dot_des, rate_max), -rate_max);

% Second-order dynamics (smooth response)
tau_rate = 0.05;  % Rate time constant [s]
tilt_ddot = (tilt_dot_des - tilt_dot) / tau_rate;

end