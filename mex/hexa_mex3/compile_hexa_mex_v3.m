%% compile_hexa_mex_v3.m - Hexarotor MPPI MEX v3 Compile Script
%
% SMC Reference Sampling version:
%   - Position SMC (fractional-order)
%   - Attitude SMC (fractional-order)
%   - Replaces PID reference rollouts
%
% MATLAB에서 실행:
%   >> compile_hexa_mex_v3
%
% 요구사항:
%   - CUDA Toolkit 설치 (11.x 또는 12.x)
%   - MATLAB Parallel Computing Toolbox
%   - nvcc가 PATH에 있어야 함

fprintf('========================================\n');
fprintf('Compiling hexa_mppi_mex_v3.cu...\n');
fprintf('========================================\n');
fprintf('SMC Reference Sampling:\n');
fprintf('  - Position SMC: s = -vel + a*e, fractional reaching\n');
fprintf('  - Attitude SMC: s = w + a*e_v + b*|e_v|^r*sign(e_v)\n');
fprintf('  - 14 SMC parameters with noise injection\n');
fprintf('========================================\n\n');

try
    % 기본 컴파일
    mexcuda -v hexa_mppi_mex_v3.cu -lcurand
    
    fprintf('\n=== Compilation successful! ===\n');
    fprintf('\nInterface:\n');
    fprintf('  [u, seq, stats] = hexa_mppi_mex_v3(x0, u_seq, u_prev, pos_des, q_des, params, mppi_params)\n');
    fprintf('\nSMC parameters in mppi_params:\n');
    fprintf('  Position: smc_pos_a(3), smc_pos_l1(3), smc_pos_l2(3), smc_pos_r\n');
    fprintf('  Attitude: smc_att_a, smc_att_b, smc_att_l1, smc_att_l2, smc_att_r\n');
    fprintf('  K_smc, sigma_smc\n');
    fprintf('\nTest command:\n');
    fprintf('  >> main_att_mppi_mex_v3\n');
    
catch ME
    fprintf('\n=== Compilation failed ===\n');
    disp(ME.message);
    
    fprintf('\nTroubleshooting:\n');
    fprintf('1. CUDA Toolkit 설치 확인: nvcc --version\n');
    fprintf('2. MATLAB GPU 지원 확인: gpuDevice\n');
    fprintf('3. CUDA 경로 설정:\n');
    fprintf('   Windows: setenv(''CUDA_PATH'', ''C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.x'')\n');
    fprintf('   Linux:   setenv(''CUDA_PATH'', ''/usr/local/cuda'')\n');
    fprintf('4. curand 라이브러리 확인\n');
    fprintf('\n대안 컴파일 (경로 명시):\n');
    fprintf('  mexcuda -v hexa_mppi_mex_v3.cu -L/usr/local/cuda/lib64 -lcurand\n');
end

%% 컴파일 옵션 (최적화 - fast_math로 powf 가속)
% 성능 최적화가 필요한 경우:
%   mexcuda -v NVCCFLAGS="-O3 -use_fast_math" hexa_mppi_mex_v3.cu -lcurand
%
% 디버그 모드:
%   mexcuda -g hexa_mppi_mex_v3.cu -lcurand