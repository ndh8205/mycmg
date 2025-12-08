function [u_opt, mppi_state] = hexa_mppi_controller_mex(x_current, mppi_state, mppi_params, params)
% hexa_mppi_controller_mex: Hexarotor MPPI (CUDA MEX wrapper)
%
% Inputs:
%   x_current   - 28x1 or 19x1 current state
%   mppi_state  - struct: .u_seq (6xN), .pos_des (3x1), .yaw_des, .omega_motor_est (6x1)
%   mppi_params - MPPI parameters
%   params      - drone parameters
%
% Outputs:
%   u_opt       - 6x1 optimal motor speed [rad/s]
%   mppi_state  - updated (shifted control sequence)
%
% 컴파일: mexcuda hexa_mppi_mex.cu -lcurand

persistent mex_compiled

if isempty(mex_compiled)
    if ~exist('hexa_mppi_mex', 'file')
        error('hexa_mppi_mex not compiled. Run: compile_hexa_mex');
    end
    mex_compiled = true;
end

%% 상태 구성 (19x1: 13 rigid body + 6 motor)
if length(x_current) >= 28
    % Full state: extract rigid body + motor speeds
    x0 = single([x_current(1:13); x_current(14:19)]);
elseif length(x_current) >= 19
    x0 = single(x_current(1:19));
elseif length(x_current) >= 13
    % Only rigid body: use estimated motor state
    if isfield(mppi_state, 'omega_motor_est')
        x0 = single([x_current(1:13); mppi_state.omega_motor_est(:)]);
    else
        error('State is 13x1 but omega_motor_est not provided in mppi_state');
    end
else
    error('State must be at least 13x1');
end
x0 = x0(:);

%% 입력 준비 (single precision)
u_seq = single(mppi_state.u_seq);    % 6xN
pos_des = single(mppi_state.pos_des(:));
yaw_des = single(mppi_state.yaw_des);

%% params struct for MEX
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

%% mppi_params struct for MEX
mex_mppi.K = int32(mppi_params.K);
mex_mppi.N = int32(mppi_params.N);
mex_mppi.dt = single(mppi_params.dt);
mex_mppi.lambda = single(mppi_params.lambda);
mex_mppi.nu = single(mppi_params.nu);
mex_mppi.sigma = single(mppi_params.sigma);
mex_mppi.w_pos_xy = single(mppi_params.w_pos_xy);
mex_mppi.w_pos_z = single(mppi_params.w_pos_z);
mex_mppi.w_vel_xy = single(mppi_params.w_vel_xy);
mex_mppi.w_vel_z = single(mppi_params.w_vel_z);
mex_mppi.w_att = single(mppi_params.w_att);
mex_mppi.w_yaw = single(mppi_params.w_yaw);
mex_mppi.w_omega_rp = single(mppi_params.w_omega_rp);
mex_mppi.w_omega_yaw = single(mppi_params.w_omega_yaw);
mex_mppi.w_terminal = single(mppi_params.w_terminal);
mex_mppi.R = single(mppi_params.R);

%% MEX 호출
[u_opt_s, u_seq_new, stats] = hexa_mppi_mex(x0, u_seq, pos_des, yaw_des, mex_params, mex_mppi);

%% 출력
u_opt = double(u_opt_s);

% Shift sequence (warm start)
N = mppi_params.N;
mppi_state.u_seq = double(u_seq_new);
mppi_state.u_seq(:, 1:N-1) = mppi_state.u_seq(:, 2:N);

% 디버깅용 저장
mppi_state.min_cost = stats.min_cost;
mppi_state.avg_cost = stats.avg_cost;
mppi_state.stats = stats;

end