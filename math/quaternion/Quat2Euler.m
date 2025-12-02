function att_euler = Quat2Euler( att_quat )

    % Quaternion to Euler angle

    % Convert quaternion to DCM
    A = GetDCM_QUAT(att_quat);

    % Extract Euler angles based on the rotation sequence (3-2-1)
    phi = atan2(A(3,2), A(3,3));         % Roll
    theta = -asin(A(3,1));               % Pitch
    psi = atan2(A(2,1), A(1,1));         % Yaw

    att_euler = [phi; theta; psi];

end  
