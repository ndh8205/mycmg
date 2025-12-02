
function q = rotation_vector_to_quat(theta)
% 회전 벡터를 쿼터니언으로 변환
    angle = norm(theta);
    if angle < 1e-10
        q = [1; 0.5*theta];
    else
        axis = theta / angle;
        q = [cos(angle/2); sin(angle/2)*axis];
    end
    q = q / norm(q);
end