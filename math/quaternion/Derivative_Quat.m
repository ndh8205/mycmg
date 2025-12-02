function dot_q = Derivative_Quat(q, omega_b) % Derivative of Quaternion - Local, Hamiltonian

    % Ensure quaternion is a column vector
    q = q(:);
    omega_b = omega_b(:);
    
    % Construct the Omega matrix based on body angular velocity
    Omega = [
                0,      -omega_b(1), -omega_b(2), -omega_b(3);
                omega_b(1),       0,  omega_b(3), -omega_b(2);
                omega_b(2), -omega_b(3),        0,  omega_b(1);
                omega_b(3),  omega_b(2), -omega_b(1),        0
            ];
    
    % Compute the quaternion derivative
    dot_q = 0.5 * Omega * q;

end