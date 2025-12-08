function [u_opt, mppi_state] = hexa_mppi_controller_mex_v3(x_current, mppi_state, mppi_params, params)
% hexa_mppi_controller_mex_v3: Hexarotor MPPI (CUDA MEX wrapper) - SMC Reference
%
% Inputs:
%   x_current   - 28x1 or 19x1 current state
%   mppi_state  - struct: .u_seq (6xN), .u_prev (6x1), .pos_des (3x1), .q_des (4x1)
%   mppi_params - MPPI parameters with SMC gains
%   params      - drone parameters
%
% Outputs:
%   u_opt       - 6x1 optimal motor speed [rad/s]
%   mppi_state  - updated (shifted control sequence, u_prev)
%
% Compile: mexcuda hexa_mppi_mex_v3.cu -lcurand

persistent mex_compiled

if isempty(mex_compiled)
    if ~exist('hexa_mppi_mex_v3', 'file')
        error('hexa_mppi_mex_v3 not compiled. Run: mexcuda hexa_mppi_mex_v3.cu -lcurand');
    end
    mex_compiled = true;
end

%% State preparation (19x1: 13 rigid body + 6 motor)
if length(x_current) >= 28
    x0 = single([x_current(1:13); x_current(14:19)]);
elseif length(x_current) >= 19
    x0 = single(x_current(1:19));
elseif length(x_current) >= 13
    if isfield(mppi_state, 'omega_motor_est')
        x0 = single([x_current(1:13); mppi_state.omega_motor_est(:)]);
    else
        error('State is 13x1 but omega_motor_est not provided in mppi_state');
    end
else
    error('State must be at least 13x1');
end
x0 = x0(:);

%% Input preparation (single precision)
u_seq = single(mppi_state.u_seq);       % 6xN
u_prev = single(mppi_state.u_prev(:));  % 6x1
pos_des = single(mppi_state.pos_des(:));
q_des = single(mppi_state.q_des(:));

%% Drone params struct for MEX
mex_params.m = single(params.drone.body.m);
mex_params.Jxx = single(params.drone.body.J(1,1));
mex_params.Jyy = single(params.drone.body.J(2,2));
mex_params.Jzz = single(params.drone.body.J(3,3));
mex_params.g = single(params.env.g);
mex_params.k_T = single(params.drone.motor.k_T);
mex_params.k_M = single(params.drone.motor.k_M);
mex_params.L = single(params.drone.body.L);
mex_params.omega_max = single(params.drone.motor.omega_b_max);
mex_params.omega_min = single(params.drone.motor.omega_b_min);
mex_params.tau_up = single(params.drone.motor.tau_up);
mex_params.tau_down = single(params.drone.motor.tau_down);

%% MPPI params struct for MEX (with SMC gains)
mex_mppi.K = int32(mppi_params.K);
mex_mppi.K_smc = int32(mppi_params.K_smc);
mex_mppi.N = int32(mppi_params.N);
mex_mppi.dt = single(mppi_params.dt);
mex_mppi.lambda = single(mppi_params.lambda);
mex_mppi.nu = single(mppi_params.nu);
mex_mppi.sigma = single(mppi_params.sigma);

% Cost weights
mex_mppi.w_pos = single(mppi_params.w_pos);
mex_mppi.w_vel = single(mppi_params.w_vel);
mex_mppi.w_att = single(mppi_params.w_att);
mex_mppi.w_omega = single(mppi_params.w_omega);
mex_mppi.w_terminal = single(mppi_params.w_terminal);
mex_mppi.w_smooth = single(mppi_params.w_smooth);
mex_mppi.R = single(mppi_params.R);

% Crash detection
mex_mppi.crash_cost = single(mppi_params.crash_cost);
mex_mppi.crash_angle = single(mppi_params.crash_angle);

% SMC Position gains
mex_mppi.smc_pos_a = double(mppi_params.smc_pos_a(:));
mex_mppi.smc_pos_l1 = double(mppi_params.smc_pos_l1(:));
mex_mppi.smc_pos_l2 = double(mppi_params.smc_pos_l2(:));
mex_mppi.smc_pos_r = single(mppi_params.smc_pos_r);

% SMC Attitude gains
mex_mppi.smc_att_a = single(mppi_params.smc_att_a);
mex_mppi.smc_att_b = single(mppi_params.smc_att_b);
mex_mppi.smc_att_l1 = single(mppi_params.smc_att_l1);
mex_mppi.smc_att_l2 = single(mppi_params.smc_att_l2);
mex_mppi.smc_att_r = single(mppi_params.smc_att_r);
mex_mppi.sigma_smc = single(mppi_params.sigma_smc);

%% MEX call
[u_opt_s, u_seq_new, stats] = hexa_mppi_mex_v3(...
    x0, u_seq, u_prev, pos_des, q_des, mex_params, mex_mppi);

%% Output
u_opt = double(u_opt_s);

% Shift sequence (warm start)
N = mppi_params.N;
mppi_state.u_seq = double(u_seq_new);
mppi_state.u_seq(:, 1:N-1) = mppi_state.u_seq(:, 2:N);

% Update previous control
mppi_state.u_prev = u_opt;

% Diagnostics
mppi_state.min_cost = stats.min_cost;
mppi_state.avg_cost = stats.avg_cost;
mppi_state.stats = stats;

end