function [tau_cmd, ctrl_state] = attitude_pid(q_des, q, omega, ctrl_state, gains, dt)
% attitude_pid: Quaternion-based PID attitude controller
%
% Inputs:
%   q_des      - 4x1 desired quaternion [qw; qx; qy; qz]
%   q          - 4x1 current quaternion
%   omega      - 3x1 angular velocity [p; q; r] [rad/s] (body frame)
%   ctrl_state - struct with integrator states
%   gains      - struct with PID gains (Kp, Ki, Kd: 3x1 or 3x3)
%   dt         - time step [s]
%
% Outputs:
%   tau_cmd    - 3x1 torque command [Nm]
%   ctrl_state - updated integrator states

%% Parameter extraction
q_des = q_des(:);
q = q(:);
omega = omega(:);

Kp = gains.Kp(:);
Ki = gains.Ki(:);
Kd = gains.Kd(:);
int_limit = gains.int_limit(:);

%% Quaternion error (body frame)
%  q_des = q ⊗ q_err  →  q_err = q^{-1} ⊗ q_des
%  Represents rotation from current to desired in body frame
q_inv = Quaternion_Conj(q);
q_err = q2q_mult(q_inv(:), q_des);

% Ensure shortest path (q and -q represent same rotation)
if q_err(1) < 0
    q_err = -q_err;
end

%% Attitude error vector
%  Small angle approximation: q_err ≈ [1; 0.5*e_att]
%  Therefore: e_att ≈ 2 * q_err(2:4)
e_att = 2 * q_err(2:4);

%% Angular rate error
%  Desired angular rate is zero for attitude hold
omega_des = zeros(3,1);
e_rate = omega_des - omega;

%% Integrator with anti-windup
ctrl_state.int_att = ctrl_state.int_att + e_att * dt;
ctrl_state.int_att = max(min(ctrl_state.int_att, int_limit), -int_limit);

%% PID control law
tau_cmd = Kp .* e_att + Ki .* ctrl_state.int_att + Kd .* e_rate;

end