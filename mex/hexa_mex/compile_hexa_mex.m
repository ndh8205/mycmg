%% compile_hexa_mex.m - Hexarotor MPPI MEX 컴파일
%
% MATLAB에서 실행:
%   >> compile_hexa_mex
%
% 요구사항:
%   - CUDA Toolkit 설치 (11.x 또는 12.x)
%   - MATLAB Parallel Computing Toolbox
%   - nvcc가 PATH에 있어야 함

fprintf('Compiling hexa_mppi_mex.cu...\n');
fprintf('CUDA Toolkit 및 curand 라이브러리 필요\n\n');

try
    % 기본 컴파일
    mexcuda -v hexa_mppi_mex.cu -lcurand
    
    fprintf('\n=== Compilation successful! ===\n');
    fprintf('Test command:\n');
    fprintf('  [u, seq, min_cost, avg_cost] = hexa_mppi_mex(x0, u_seq, pos_des, yaw_des, params, mppi_params)\n');
    
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
    fprintf('  mexcuda -v hexa_mppi_mex.cu -L/usr/local/cuda/lib64 -lcurand\n');
end

%% 컴파일 옵션 (최적화)
% 성능 최적화가 필요한 경우:
%   mexcuda -v NVCCFLAGS="-O3 -use_fast_math" hexa_mppi_mex.cu -lcurand
%
% 디버그 모드:
%   mexcuda -g hexa_mppi_mex.cu -lcurand