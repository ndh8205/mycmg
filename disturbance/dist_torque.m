function [tau_dist, dist_state] = dist_torque(t, params, dist_state)
% dist_torque: Calculate torque disturbance
%
% Inputs:
%   t          - current time [s]
%   params     - parameter struct with dist field
%   dist_state - disturbance state struct
%
% Outputs:
%   tau_dist   - 3x1 torque disturbance [Nm] (body frame)
%   dist_state - updated state

tau_dist = zeros(3,1);

if ~params.dist.enable
    return;
end

mag = params.dist.torque.magnitude(:);
dist_type = params.dist.torque.type;

switch dist_type
    case 'none'
        tau_dist = zeros(3,1);
        
    case 'sine'
        % Single frequency sinusoidal
        f = params.dist.torque.freq;
        tau_dist = mag .* sin(2*pi*f*t);
        
    case 'random_sine'
        % Sum of multiple sinusoids with random phases
        tau_dist = zeros(3,1);
        f_base = params.dist.torque.freq;
        for i = 1:3
            for j = 1:5
                f_j = f_base * dist_state.random_sine.freq_mult(i,j) * j;
                phi_j = dist_state.random_sine.phase(i,j);
                tau_dist(i) = tau_dist(i) + sin(2*pi*f_j*t + phi_j) / j;
            end
        end
        tau_dist = mag .* tau_dist / 3;  % Normalize
        
    case 'step'
        % Step disturbance
        t_step = params.dist.torque.step_time;
        t_dur = params.dist.torque.step_duration;
        if t >= t_step && t < t_step + t_dur
            tau_dist = mag;
        end
        
    case 'impulse'
        % Short impulse
        t_imp = params.dist.torque.impulse_time;
        t_dur = params.dist.torque.impulse_duration;
        if t >= t_imp && t < t_imp + t_dur
            tau_dist = mag;
        end
        
    case 'combined'
        % Combination: random_sine + step
        % Random sine component
        tau_sine = zeros(3,1);
        f_base = params.dist.torque.freq;
        for i = 1:3
            for j = 1:5
                f_j = f_base * dist_state.random_sine.freq_mult(i,j) * j;
                phi_j = dist_state.random_sine.phase(i,j);
                tau_sine(i) = tau_sine(i) + sin(2*pi*f_j*t + phi_j) / j;
            end
        end
        tau_sine = mag .* tau_sine / 3;
        
        % Step component (50% magnitude)
        tau_step = zeros(3,1);
        t_step = params.dist.torque.step_time;
        t_dur = params.dist.torque.step_duration;
        if t >= t_step && t < t_step + t_dur
            tau_step = 0.5 * mag;
        end
        
        tau_dist = tau_sine + tau_step;
        
    case 'paper'
        % Paper-style: integrated random sine (from reference paper)
        % Multiple sinusoids summed and integrated for drift-like behavior
        
        accel_size = params.dist.torque.paper.accel_size(:);
        scale = params.dist.torque.paper.scale;
        max_tau = params.dist.torque.paper.max_torque;
        
        % Time step from previous call
        dt = t - dist_state.t_prev;
        if dt <= 0
            dt = 0.001;
        end
        
        % Compute acceleration: sum of multiple sinusoids per axis
        accel = zeros(3,1);
        for i = 1:3
            for j = 1:size(dist_state.paper.freq, 2)
                f_j = dist_state.paper.freq(i,j);
                phi_j = dist_state.paper.phase(i,j);
                amp_j = dist_state.paper.amp(i,j);
                accel(i) = accel(i) + amp_j * sin(2*pi*f_j*t + phi_j);
            end
        end
        accel = accel_size .* accel;
        
        % Integrate (Euler) - this creates drift-like behavior
        dist_state.paper.integral = dist_state.paper.integral + dt * accel * 2;
        
        % Scale and clamp to bounded range
        tau_dist = dist_state.paper.integral / scale;
        tau_dist = max(min(tau_dist, max_tau), -max_tau);
        
    otherwise
        tau_dist = zeros(3,1);
end

% Update state
dist_state.t_prev = t;

end