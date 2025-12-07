function [u_opt, mppi_state] = cmg_mppi_controller_mex(x_current, mppi_state, mppi_params, params)
% cmg_mppi_controller_mex: CMG MPPI (CUDA MEX wrapper)
%
% 사전에 mexcuda cmg_mppi_mex.cu 컴파일 필요

persistent mex_compiled

if isempty(mex_compiled)
    % MEX 파일 존재 확인
    if ~exist('cmg_mppi_mex', 'file')
        error('cmg_mppi_mex not compiled. Run: mexcuda cmg_mppi_mex.cu');
    end
    mex_compiled = true;
end

%% 입력 준비 (single precision, column-major)
x0 = single(x_current(:));
u_seq = single(mppi_state.u_seq);  % 2xN
att_des = single(mppi_state.att_des(:));

%% params struct for MEX
mex_params.Jxx = single(params.J(1,1));
mex_params.Jyy = single(params.J(2,2));
mex_params.Jzz = single(params.J(3,3));
mex_params.h1 = single(params.h1);
mex_params.h2 = single(params.h2);
mex_params.gr_max = single(params.gimbal_rate_max);

%% mppi_params struct for MEX
mex_mppi.K = int32(mppi_params.K);
mex_mppi.N = int32(mppi_params.N);
mex_mppi.dt = single(mppi_params.dt);
mex_mppi.lambda = single(mppi_params.lambda);
mex_mppi.nu = single(mppi_params.nu);
mex_mppi.sigma1 = single(mppi_params.sigma(1) / sqrt(mppi_params.dt));
mex_mppi.sigma2 = single(mppi_params.sigma(2) / sqrt(mppi_params.dt));
mex_mppi.R1 = single(mppi_params.R(1));
mex_mppi.R2 = single(mppi_params.R(2));
mex_mppi.Q2 = single(mppi_params.Q_att(2));
mex_mppi.Q3 = single(mppi_params.Q_att(3));
mex_mppi.Qw = single(mppi_params.Q_omega(1));
mex_mppi.Sw = single(mppi_params.S_weight);
mex_mppi.Wt = single(mppi_params.w_terminal);

%% MEX 호출
[u_opt, u_seq_new] = cmg_mppi_mex(x0, u_seq, att_des, mex_params, mex_mppi);

%% 출력
u_opt = double(u_opt);

% Shift sequence
N = mppi_params.N;
mppi_state.u_seq = double(u_seq_new);
mppi_state.u_seq(:, 1:N-1) = mppi_state.u_seq(:, 2:N);

end