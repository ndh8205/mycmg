function omega_m_dot = motor_dynamics(omega_m, u_cmd, params)
% motor_dynamics: First-order motor response model
%
% Inputs:
%   omega_m - 4x1 current motor angular velocity [rad/s]
%   u_cmd   - 4x1 commanded motor angular velocity [rad/s]
%   params  - parameter struct
%
% Output:
%   omega_m_dot - 4x1 motor angular velocity derivative [rad/s^2]
%
% Model:
%   tau * d(omega_m)/dt + omega_m = u_cmd
%   where tau = tau_up (accelerating) or tau_down (decelerating)

%% Parameter extraction
tau_up   = params.drone.motor.tau_up;
tau_down = params.drone.motor.tau_down;
omega_max = params.drone.motor.omega_b_max;
omega_min = params.drone.motor.omega_b_min;

%% Saturate command
u_sat = max(min(u_cmd, omega_max), omega_min);

%% First-order dynamics with asymmetric time constant
omega_m_dot = zeros(4,1);
for i = 1:4
    if u_sat(i) >= omega_m(i)
        tau = tau_up;    % Accelerating
    else
        tau = tau_down;  % Decelerating
    end
    omega_m_dot(i) = (u_sat(i) - omega_m(i)) / tau;
end

end