function [x_dot, varargout] = tiltrotor_dynamics(x, u, w, dt, params, k)
% tiltrotor_dynamics: 6-DoF tiltrotor dynamics model
%
% State vector x (30x1):
%   x(1:3)   - position [m] (NED frame)
%   x(4:6)   - velocity [m/s] (body frame)
%   x(7:10)  - quaternion [qw,qx,qy,qz] (body to NED)
%   x(11:13) - angular velocity [rad/s] (body frame)
%   x(14:17) - motor angular velocity [rad/s]
%   x(18:20) - gyro bias [rad/s]
%   x(21:23) - accelerometer bias [m/s^2]
%   x(24:26) - magnetometer bias [Gauss]
%   x(27:28) - tilt angles [rad] (rotor 0, 2)
%   x(29:30) - tilt rates [rad/s]
%
% Input u (9x1):
%   u(1:4) - motor speed command [rad/s]
%   u(5:6) - tilt angle command [rad]
%   u(7:8) - elevon command [rad] (for future)
%   u(9)   - elevator command [rad] (for future)
%
% Process noise w (9x1):
%   w(1:3) - gyro bias noise
%   w(4:6) - accel bias noise
%   w(7:9) - mag bias noise

%% Extract states
pos      = x(1:3);
vel_b    = x(4:6);
quat     = x(7:10);
omega    = x(11:13);
omega_m  = x(14:17);
b_gyro   = x(18:20);
b_accel  = x(21:23);
b_mag    = x(24:26);
tilt     = x(27:28);   % [tilt_0; tilt_2]
tilt_dot = x(29:30);

%% Extract inputs
u_motor = u(1:4);
u_tilt  = u(5:6);
% u_elevon = u(7:8);  % Future use
% u_elev   = u(9);    % Future use

%% Parameters
m   = params.drone.body.m;
J   = params.drone.body.J;
g   = params.env.g;
rho = params.env.rho;

k_T = params.drone.motor.k_T;
k_M = params.drone.motor.k_M;

rotor_pos = params.drone.rotor.pos;     % 4x3
rotor_dir = params.drone.rotor.dir;     % 4x1 (CCW=+1, CW=-1)
tiltable  = params.drone.rotor.tiltable; % 4x1

%% Normalize quaternion
quat = quat / norm(quat);

%% DCM (body to NED)
R_b2n = GetDCM_QUAT(quat);
R_n2b = R_b2n';

%% Airspeed (body frame)
vel_air_b = vel_b;  % Assuming no wind for now
V_air = norm(vel_air_b);

%% ========== ROTOR FORCES AND MOMENTS ==========
F_rotor_total = zeros(3,1);
M_rotor_total = zeros(3,1);

% Build full tilt array (4 rotors, only 0 and 2 tilt)
tilt_full = zeros(4,1);
tilt_full(1) = tilt(1);  % rotor 0
tilt_full(3) = tilt(2);  % rotor 2

for i = 1:4
    % Thrust magnitude
    T_i = k_T * omega_m(i)^2;
    
    % Thrust direction in body frame
    % tilt=0: thrust along -Z (up in NED)
    % tilt>0: thrust tilts forward (+X)
    if tiltable(i)
        tilt_i = tilt_full(i);
        F_dir = [sin(tilt_i); 0; -cos(tilt_i)];
    else
        F_dir = [0; 0; -1];
    end
    
    F_i = T_i * F_dir;
    
    % Rotor position
    r_i = rotor_pos(i, :)';
    
    % Moment from thrust
    M_thrust_i = cross(r_i, F_i);
    
    % Reaction torque (about rotor axis)
    % CCW (dir=+1): positive yaw, CW (dir=-1): negative yaw
    tau_reaction = k_M * T_i * rotor_dir(i);
    if tiltable(i)
        tilt_i = tilt_full(i);
        % Reaction torque direction: [0;0;1] rotated by tilt about Y
        % At tilt=0: [0;0;1], at tilt=90: [-1;0;0]
        M_reaction_i = tau_reaction * [sin(tilt_i); 0; cos(tilt_i)];
    else
        M_reaction_i = [0; 0; tau_reaction];
    end
    
    % Sum
    F_rotor_total = F_rotor_total + F_i;
    M_rotor_total = M_rotor_total + M_thrust_i + M_reaction_i;
end

%% ========== AERODYNAMIC FORCES AND MOMENTS ==========
F_aero = zeros(3,1);
M_aero = zeros(3,1);

if V_air > 1.0  % Only compute aero if sufficient airspeed
    % Control surface inputs [left_elevon; right_elevon; elevator]
    u_ctrl = [u(7); u(8); u(9)];
    [F_aero, M_aero] = aero_model(vel_air_b, omega, u_ctrl, params);
end

%% ========== TOTAL FORCES AND MOMENTS ==========
F_body = F_rotor_total + F_aero;
M_body = M_rotor_total + M_aero;

%% ========== EQUATIONS OF MOTION ==========

% Position derivative (NED = R_b2n * vel_body)
pos_dot = R_b2n * vel_b;

% Velocity derivative (body frame)
gravity_body = R_n2b * [0; 0; g];
vel_dot = F_body / m + gravity_body - skew3(omega) * vel_b;

% Quaternion derivative
quat_dot = Derivative_Quat(quat, omega);

% Angular velocity derivative (Euler's equation)
omega_dot = J \ (M_body - skew3(omega) * (J * omega));

% Motor dynamics
omega_m_dot = motor_dynamics(omega_m, u_motor, params);

% Tilt actuator dynamics (first-order with rate limit)
tilt_ddot = tilt_actuator_dynamics(tilt, tilt_dot, u_tilt, params);

% Bias dynamics
b = [b_gyro; b_accel; b_mag];
b_dot = bias_dynamics(b, w, params);

%% ========== GROUND COLLISION ==========
if pos(3) >= 0
    pos_dot = zeros(3,1);
    vel_dot = zeros(3,1);
    omega_dot = zeros(3,1);
end

%% ========== OUTPUT ==========
x_dot = [pos_dot; vel_dot; quat_dot; omega_dot; omega_m_dot; b_dot; tilt_dot; tilt_ddot];

% Optional outputs for diagnostics
if nargout > 1
    varargout{1} = norm(F_rotor_total);  % Total thrust
    varargout{2} = tilt;                  % Tilt angles
    varargout{3} = V_air;                 % Airspeed
    varargout{4} = F_aero;                % Aero forces
    varargout{5} = M_aero;                % Aero moments
    varargout{6} = M_body;                % Total moment
end

end  % End of main function

%% ========== LOCAL FUNCTIONS ==========

function tilt_ddot = tilt_actuator_dynamics(tilt, tilt_dot, u_tilt, params)
% First-order servo dynamics with rate limiting
    
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
    tau_rate = 0.05;  % Rate time constant
    tilt_ddot = (tilt_dot_des - tilt_dot) / tau_rate;
end

function [F_aero, M_aero] = aero_forces(vel_air_b, omega, quat, u, params)
% Aerodynamic force and moment calculation
% Simple drag model for now

    F_aero = zeros(3,1);
    M_aero = zeros(3,1);
    
    rho = params.env.rho;
    V = norm(vel_air_b);
    
    if V > 0.1
        % Simple body drag
        CD_body = 0.5;
        A_ref = 0.1;  % Reference area [m^2]
        
        D = 0.5 * rho * V^2 * CD_body * A_ref;
        F_aero = -D * (vel_air_b / V);
    end
end