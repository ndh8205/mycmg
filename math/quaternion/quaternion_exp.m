function q_exp = quaternion_exp(omega_vec, dt)
    % Quaternion exponential for rotation vector
    % omega_vec: 3x1 angular velocity vector [rad/s]
    % dt: time step [s]
    % q_exp: 4x1 quaternion [qw; qx; qy; qz]
    
    omega_vec = omega_vec(:); % 열벡터로 보장
    
    % 회전각과 축 계산
    angle = norm(omega_vec) * dt;
    
    if angle < 1e-8
        % 작은 각도 근사
        q_exp = [1; 0.5*omega_vec*dt];
    else
        % 일반적인 경우
        axis = omega_vec / norm(omega_vec);
        q_exp = [cos(angle/2); sin(angle/2) * axis];
    end
    
    % 정규화
    q_exp = q_exp / norm(q_exp);
end