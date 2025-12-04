function params_true = apply_uncertainty(params)
% apply_uncertainty: Apply parameter uncertainty to create "true" system
%
% Input:
%   params - nominal parameter struct
%
% Output:
%   params_true - parameters with uncertainty applied (for simulation)
%
% Usage:
%   params_nom = params_init('quad');                % Nominal (for controller)
%   [params_nom, dist_state] = dist_init(params_nom, 'level3');
%   params_true = apply_uncertainty(params_nom);     % True (for dynamics)
%
%   Controller uses: params_nom
%   Dynamics uses:   params_true

% Copy all parameters
params_true = params;

if ~isfield(params, 'dist') || ~params.dist.uncertainty.enable
    return;
end

% Set random seed for reproducibility (optional)
% rng(42);

%% Mass uncertainty
if params.dist.uncertainty.mass > 0
    delta_m = params.dist.uncertainty.mass;
    scale = 1 + delta_m * (2*rand() - 1);  % Uniform in [1-delta, 1+delta]
    params_true.drone.body.m = params.drone.body.m * scale;
end

%% Inertia uncertainty
if params.dist.uncertainty.inertia > 0
    delta_J = params.dist.uncertainty.inertia;
    J_nom = params.drone.body.J;
    
    % Apply uncertainty to diagonal elements
    scale_xx = 1 + delta_J * (2*rand() - 1);
    scale_yy = 1 + delta_J * (2*rand() - 1);
    scale_zz = 1 + delta_J * (2*rand() - 1);
    
    params_true.drone.body.J = diag([J_nom(1,1)*scale_xx, ...
                                     J_nom(2,2)*scale_yy, ...
                                     J_nom(3,3)*scale_zz]);
end

%% Thrust coefficient uncertainty
if params.dist.uncertainty.k_T > 0
    delta_kT = params.dist.uncertainty.k_T;
    scale = 1 + delta_kT * (2*rand() - 1);
    params_true.drone.motor.k_T = params.drone.motor.k_T * scale;
end

%% Print uncertainty info
fprintf('=== Applied Parameter Uncertainty ===\n');
fprintf('Mass:    %.3f kg (nominal: %.3f kg, %.1f%%)\n', ...
    params_true.drone.body.m, params.drone.body.m, ...
    100*(params_true.drone.body.m/params.drone.body.m - 1));
fprintf('Jxx:     %.6f (nominal: %.6f, %.1f%%)\n', ...
    params_true.drone.body.J(1,1), params.drone.body.J(1,1), ...
    100*(params_true.drone.body.J(1,1)/params.drone.body.J(1,1) - 1));
fprintf('Jyy:     %.6f (nominal: %.6f, %.1f%%)\n', ...
    params_true.drone.body.J(2,2), params.drone.body.J(2,2), ...
    100*(params_true.drone.body.J(2,2)/params.drone.body.J(2,2) - 1));
fprintf('Jzz:     %.6f (nominal: %.6f, %.1f%%)\n', ...
    params_true.drone.body.J(3,3), params.drone.body.J(3,3), ...
    100*(params_true.drone.body.J(3,3)/params.drone.body.J(3,3) - 1));
fprintf('k_T:     %.6e (nominal: %.6e, %.1f%%)\n', ...
    params_true.drone.motor.k_T, params.drone.motor.k_T, ...
    100*(params_true.drone.motor.k_T/params.drone.motor.k_T - 1));
fprintf('=====================================\n');

end