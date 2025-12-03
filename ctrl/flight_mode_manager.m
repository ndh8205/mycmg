function [tilt_cmd, mode, blend] = flight_mode_manager(V_air, params)
% flight_mode_manager: Tiltrotor flight mode and tilt scheduling
%
% Inputs:
%   V_air  - airspeed [m/s]
%   params - parameter struct
%
% Outputs:
%   tilt_cmd - 2x1 tilt angle command [rad] (front-right, front-left)
%   mode     - flight mode (0=MC, 1=transition, 2=FW)
%   blend    - blending factor [0,1] (0=full MC, 1=full FW)
%
% Flight Modes:
%   MC (Multicopter): tilt = 0, rotor thrust for control
%   Transition: tilt varies with airspeed
%   FW (Fixed-Wing): tilt = 90°, aerodynamic surfaces for control

%% Parameters
V_start = params.flight.V_transition_start;  % [m/s]
V_end   = params.flight.V_transition_end;    % [m/s]
tilt_max = params.drone.tilt.max;            % [rad] (~90°)

%% Mode determination and blending
if V_air < V_start
    % Multicopter mode
    mode = 0;
    blend = 0;
    tilt_angle = 0;
    
elseif V_air > V_end
    % Fixed-wing mode
    mode = 2;
    blend = 1;
    tilt_angle = tilt_max;
    
else
    % Transition mode
    mode = 1;
    
    % Linear blending
    blend = (V_air - V_start) / (V_end - V_start);
    
    % Smooth tilt schedule (sinusoidal for smoother transition)
    % tilt = tilt_max * sin(blend * pi/2)^2
    tilt_angle = tilt_max * sin(blend * pi/2)^2;
end

%% Output (same tilt for both front rotors)
tilt_cmd = [tilt_angle; tilt_angle];

end