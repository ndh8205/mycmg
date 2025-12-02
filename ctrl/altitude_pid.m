function [thrust_cmd, ctrl_state] = altitude_pid(alt_des, alt, vel_z, ctrl_state, gains, dt, params)
% altitude_pid: PID altitude controller
%
% Inputs:
%   alt_des    - desired altitude [m] (positive up)
%   alt        - current altitude [m] (positive up)
%   vel_z      - vertical velocity [m/s] (positive up)
%   ctrl_state - struct with integrator state
%   gains      - struct with PID gains
%   dt         - time step [s]
%   params     - system parameters
%
% Outputs:
%   thrust_cmd - total thrust command [N]
%   ctrl_state - updated integrator state

% Altitude error
e_alt = alt_des - alt;

% Derivative
e_dot = -vel_z;

% Integrator update with anti-windup
ctrl_state.int_alt = ctrl_state.int_alt + e_alt * dt;
int_limit = gains.int_limit;
ctrl_state.int_alt = max(min(ctrl_state.int_alt, int_limit), -int_limit);

% PID output (acceleration command)
acc_cmd = gains.Kp * e_alt + gains.Ki * ctrl_state.int_alt + gains.Kd * e_dot;

% Convert to thrust: T = m * (g + acc_cmd)
m = params.drone.body.m;
g = params.env.g;
thrust_cmd = m * (g + acc_cmd);

% Thrust saturation
thrust_max = 4 * params.drone.motor.k_T * params.drone.motor.omega_b_max^2;
thrust_min = 0;
thrust_cmd = max(min(thrust_cmd, thrust_max), thrust_min);

end
