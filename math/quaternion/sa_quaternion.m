function delta_quat = sa_quaternion(dtheta)

    % dtheta: 3x1
    % delta_quat: 4x1
    th = 0.5 * dtheta;
    delta_quat = [1; th];
    delta_quat = delta_quat / norm(delta_quat);
    
end
