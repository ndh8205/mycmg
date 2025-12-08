%% compile_hexa_mex_v2.m - Hexarotor MPPI MEX v2 Compile Script
%
% Improved cost function version:
%   - Quadratic attitude cost (replaces log barrier)
%   - Smoothness penalty
%   - Crash detection
%   - Simplified parameters
%
% MATLAB에서 실행:
%   >> compile_hexa_mex_v2
%
% 요구사항:
%   - CUDA Toolkit 설치 (11.x 또는 12.x)
%   - MATLAB Parallel Computing Toolbox
%   - nvcc가 PATH에 있어야 함

fprintf('========================================\n');
fprintf('Compiling hexa_mppi_mex_v2.cu...\n');
fprintf('========================================\n');
fprintf('Improved cost function:\n');
fprintf('  - Quadratic attitude cost (quaternion error)\n');
fprintf('  - Control smoothness penalty\n');
fprintf('  - Crash detection & penalty\n');
fprintf('  - Simplified parameters (9 vs 12)\n');
fprintf('========================================\n\n');

try
    % 기본 컴파일
    mexcuda -v hexa_mppi_mex_v2.cu -lcurand
    
    fprintf('\n=== Compilation successful! ===\n');
    fprintf('\nNew interface:\n');
    fprintf('  [u, seq, stats] = hexa_mppi_mex_v2(x0, u_seq, u_prev, pos_des, q_des, params, mppi_params)\n');
    fprintf('\nNew inputs:\n');
    fprintf('  u_prev  - 6x1 previous control (for smoothness)\n');
    fprintf('  q_des   - 4x1 desired quaternion (for attitude tracking)\n');
    fprintf('\nSimplified mppi_params:\n');
    fprintf('  w_pos, w_vel, w_att, w_omega, w_terminal, w_smooth, R\n');
    fprintf('  crash_cost, crash_angle\n');
    fprintf('\nTest command:\n');
    fprintf('  >> main_att_mppi_mex_v2\n');
    
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
    fprintf('  mexcuda -v hexa_mppi_mex_v2.cu -L/usr/local/cuda/lib64 -lcurand\n');
end

%% 컴파일 옵션 (최적화)
% 성능 최적화가 필요한 경우:
%   mexcuda -v NVCCFLAGS="-O3 -use_fast_math" hexa_mppi_mex_v2.cu -lcurand
%
% 디버그 모드:
%   mexcuda -g hexa_mppi_mex_v2.cu -lcurand