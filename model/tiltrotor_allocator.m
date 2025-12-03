function [output, B] = tiltrotor_allocator(input, tilt_angles, params, mode)
% tiltrotor_allocator: Quad-Tiltrotor control allocation based on standard_vtol.sdf
%
% Usage:
%   [omega_sq, B] = tiltrotor_allocator(cmd_vec, tilt_angles, params, 'inverse')
%   [cmd_vec, B]  = tiltrotor_allocator(omega_sq, tilt_angles, params, 'forward')
%
% Inputs:
%   input       - 4x1 vector
%                 'inverse': [T; τx; τy; τz] (Thrust, Roll, Pitch, Yaw)
%                 'forward': [ω0²; ω1²; ω2²; ω3²] (Squared motor speeds)
%   tilt_angles - 2x1 vector [alpha_0; alpha_2] (Front Right, Front Left) [rad]
%                 (0 rad = Upward Thrust, >0 rad = Tilted Forward)
%   params      - parameter struct
%   mode        - 'inverse' (default) or 'forward'
%
% Outputs:
%   output - 4x1 vector (Motor speeds or Forces/Moments)
%   B      - 4x4 allocation matrix (Time-varying based on tilt)
%
% Note on Coordinates (NED Frame):
%   X: Forward, Y: Right, Z: Down
%   Positive Roll: Right wing down
%   Positive Pitch: Nose up
%   Positive Yaw: Clockwise (Turn Right)

if nargin < 4
    mode = 'inverse';
end

%% 1. Parameters (Extracted from standard_vtol.sdf)
% motorConstant: 2e-05, momentConstant: 0.06
k_T = 2.0e-05;   % [N/(rad/s)^2]
k_Q_ratio = 0.06; 
k_M = k_T * k_Q_ratio; % [N*m/(rad/s)^2] Drag Moment Coeff

%% 2. Geometry & Configuration
% SDF Position: 0.35, -0.35 (ENU) -> NED Conversion needed
% In NED: X=Forward, Y=Right.
% Motor 0 (Front-Right): x = +0.35, y = +0.35
% Motor 1 (Back-Left):   x = -0.35, y = -0.35
% Motor 2 (Front-Left):  x = +0.35, y = -0.35
% Motor 3 (Back-Right):  x = -0.35, y = +0.35

L_x = 0.35; % [m]
L_y = 0.35; % [m]

% Tilt Angles (Standard VTOL allows front motors 0 & 2 to tilt)
a0 = tilt_angles(1); % Motor 0 (Front-Right)
a2 = tilt_angles(2); % Motor 2 (Front-Left)
% Rear motors are fixed (alpha = 0)
a1 = 0;
a3 = 0;

% Trig values
c0 = cos(a0); s0 = sin(a0);
c2 = cos(a2); s2 = sin(a2);

%% 3. Allocation Matrix B Construction
% We map the SDF motor order to the column vectors:
% Col 1: Motor 0 (Front-Right, CCW, Tilted)
% Col 2: Motor 1 (Back-Left,   CCW, Fixed)
% Col 3: Motor 2 (Front-Left,  CW,  Tilted)
% Col 4: Motor 3 (Back-Right,  CW,  Fixed)

% User Convention for Torque Signs:
% Roll(τx):  Left(+), Right(-)
% Pitch(τy): Front(+), Back(-)
% Yaw(τz):   CCW(+), CW(-)  <-- Note: Standard dynamics usually say CCW prop makes CW torque(-)

% --- Derivation of Matrix Elements ---
% Thrust (T_z): -Sum(T * cos(alpha)) -> But user wants "Thrust magnitude" usually.
% Let's assume input T is total vertical thrust command.
% Element = k_T * cos(alpha)

% Roll (τx): Force_Z * Arm_Y
% M0(FR): Right(-). Eff_Thrust = T*c0.  -> -L_y * k_T * c0
% M1(BL): Left(+).  Eff_Thrust = T*1.   -> +L_y * k_T
% M2(FL): Left(+).  Eff_Thrust = T*c2.  -> +L_y * k_T * c2
% M3(BR): Right(-). Eff_Thrust = T*1.   -> -L_y * k_T

% Pitch (τy): Force_Z * Arm_X
% M0(FR): Front(+). -> +L_x * k_T * c0
% M1(BL): Back(-).  -> -L_x * k_T
% M2(FL): Front(+). -> +L_x * k_T * c2
% M3(BR): Back(-).  -> -L_x * k_T

% Yaw (τz): Drag Torque + Thrust Vectoring
% Component 1: Drag Torque (Projected on Z axis)
%   - CCW props (0,1): Create Positive Yaw (per user convention) -> +k_M * cos(alpha)
%   - CW props (2,3): Create Negative Yaw -> -k_M * cos(alpha)
% Component 2: Thrust Vectoring (Force_X * Arm_Y)
%   - F_x = T * sin(alpha) (Forward force)
%   - M0(FR): Pulls forward on Right side -> Turns Nose Left (Negative Yaw) -> -L_y * T * s0
%   - M2(FL): Pulls forward on Left side  -> Turns Nose Right (Positive Yaw)-> +L_y * T * s2

B = zeros(4,4);

% 1. Thrust (Vertical component)
B(1,1) = k_T * c0;      % Motor 0
B(1,2) = k_T;           % Motor 1 (Fixed)
B(1,3) = k_T * c2;      % Motor 2
B(1,4) = k_T;           % Motor 3 (Fixed)

% 2. Roll (τx)
B(2,1) = -L_y * k_T * c0;
B(2,2) =  L_y * k_T;
B(2,3) =  L_y * k_T * c2;
B(2,4) = -L_y * k_T;

% 3. Pitch (τy)
B(3,1) =  L_x * k_T * c0;
B(3,2) = -L_x * k_T;
B(3,3) =  L_x * k_T * c2;
B(3,4) = -L_x * k_T;

% 4. Yaw (τz) - THE COMPLEX PART
% (Drag Torque component) + (Thrust Vectoring component)
B(4,1) = ( k_M * c0) - (L_y * k_T * s0); % Motor 0 (CCW, Right Arm)
B(4,2) = ( k_M);                         % Motor 1 (CCW, Left Arm, Fixed)
B(4,3) = (-k_M * c2) + (L_y * k_T * s2); % Motor 2 (CW, Left Arm)
B(4,4) = (-k_M);                         % Motor 3 (CW, Right Arm, Fixed)

%% 4. Compute Output
if strcmp(mode, 'inverse')
    % Check for ill-conditioning (use condition number, not determinant)
    if cond(B) > 1e6
        warning('Allocator Matrix B is ill-conditioned (cond=%.2e).', cond(B));
        output = pinv(B) * input;
    else
        output = B \ input;
    end
else
    output = B * input;
end

end