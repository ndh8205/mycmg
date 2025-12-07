%% compile_mex.m - CMG MPPI MEX 컴파일
%
% MATLAB에서 실행:
%   >> compile_mex
%
% 요구사항:
%   - CUDA Toolkit 설치
%   - MATLAB Parallel Computing Toolbox
%   - nvcc가 PATH에 있어야 함

fprintf('Compiling cmg_mppi_mex.cu...\n');

try
    mexcuda -v cmg_mppi_mex.cu -lcurand
    fprintf('Compilation successful!\n');
    fprintf('Test: [u, seq] = cmg_mppi_mex(x0, u_seq, att_des, params, mppi_params)\n');
catch ME
    fprintf('Compilation failed:\n');
    disp(ME.message);
    fprintf('\nTroubleshooting:\n');
    fprintf('1. Check CUDA Toolkit installed: nvcc --version\n');
    fprintf('2. Check MATLAB GPU support: gpuDevice\n');
    fprintf('3. Set CUDA path: setenv(''CUDA_PATH'', ''C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.0'')\n');
end