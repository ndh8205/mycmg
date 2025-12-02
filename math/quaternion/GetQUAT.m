function quat = GetQUAT( psi, theta, phi )  % Euler to Quaternion (Roll, Pitch, Yaw)

    % phi: Roll (radians)
    % theta: Pitch (radians)
    % psi: Yaw (radians)

    cPhi = cos(phi/2);
    sPhi = sin(phi/2);
    cTheta = cos(theta/2);
    sTheta = sin(theta/2);
    cPsi = cos(psi/2);
    sPsi = sin(psi/2);

    qw = cPhi*cTheta*cPsi + sPhi*sTheta*sPsi;
    qx = sPhi*cTheta*cPsi - cPhi*sTheta*sPsi;
    qy = cPhi*sTheta*cPsi + sPhi*cTheta*sPsi;
    qz = cPhi*cTheta*sPsi - sPhi*sTheta*cPsi;

    quat = [qw, qx, qy, qz];
end
