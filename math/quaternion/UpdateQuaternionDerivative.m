function q_dot = UpdateQuaternionDerivative(q, omega_A, omega_B)

    % PDF p.14의 상대 쿼터니언 미분 방정식을 해밀토니언으로 바꿈
    % q̇_B2A = (1/2)Θ( ω_A , ω_B)q
    % where Θ(ω_A, ω_B) = Ω(ω_B) - Γ(ω_A)
    %
    % Inputs:
    %   q       - 상대 쿼터니언 q_B2A (4x1)
    %   omega_A - Frame A의 각속도 (A frame 표현) [rad/s]
    %   omega_B - Frame B의 각속도 (B frame 표현) [rad/s]
    %
    % Output:
    %   q_dot   - 쿼터니언 시간 미분 (4x1) - q̇_B2A
    
    Theta = GetThetaMatrix(omega_A, omega_B);
    q_dot = 0.5 * Theta * q(:);
end