function Theta = GetThetaMatrix(omega_A, omega_B)
% Theta 행렬 계산 (상대 쿼터니언 미분용)
%
% Inputs:
%   omega_d - Deputy 각속도 (3x1, B frame)
%   omega_c - Chief 각속도 (3x1, A frame)
%
% Output:
%   Theta - 4x4 행렬
% 
% 개념 : Local, Global -> Joan Sola 논문에서 사용하는 개념

omega_A = omega_A(:); % Global
omega_B = omega_B(:); % Local

gamma = GetGammaMatrix(omega_A); % Left mult.
omega = GetOmegaMatrix(omega_B); % Right mult.

Theta = omega - gamma;

end
