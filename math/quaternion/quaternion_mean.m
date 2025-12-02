function q_mean = quaternion_mean(Q, w)
% 가중 쿼터니언 평균
    [~, n] = size(Q);
    
    % 첫 번째를 기준으로
    q_mean = Q(:,1);
    
    % 반복적 평균
    for iter = 1:10
        e = zeros(3,1);
        for i = 1:n
            q_i = Q(:,i);
            % 상대 회전
            dq = q2q_mult(q_i, inv_q(q_mean));
            if dq(1) < 0, dq = -dq; end
            % 작은 회전 벡터
            e = e + w(i) * 2 * dq(2:4);
        end
        
        % 업데이트
        theta = norm(e);
        if theta > 1e-10
            axis = e/theta;
            dq_update = [cos(theta/2); sin(theta/2)*axis];
        else
            dq_update = [1; 0.5*e];
        end
        q_mean = q2q_mult(dq_update, q_mean);
        q_mean = q_mean / norm(q_mean);
        
        if norm(e) < 1e-10
            break;
        end
    end
end