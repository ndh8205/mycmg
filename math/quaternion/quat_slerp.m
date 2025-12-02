
%% ========== 보조 함수: SLERP ==========
function q_interp = quat_slerp(q1, q2, t)
    % Spherical Linear Interpolation
    q1 = q1(:) / norm(q1);
    q2 = q2(:) / norm(q2);
    
    % 내적으로 각도 계산
    dot_prod = q1' * q2;
    
    % 최단 경로 선택
    if dot_prod < 0
        q2 = -q2;
        dot_prod = -dot_prod;
    end
    
    % 거의 같은 경우
    if dot_prod > 0.9995
        q_interp = q1 + t*(q2 - q1);
        q_interp = q_interp / norm(q_interp);
        return;
    end
    
    % SLERP
    theta = acos(dot_prod);
    sin_theta = sin(theta);
    
    w1 = sin((1-t)*theta) / sin_theta;
    w2 = sin(t*theta) / sin_theta;
    
    q_interp = w1*q1 + w2*q2;
    q_interp = q_interp / norm(q_interp);
end