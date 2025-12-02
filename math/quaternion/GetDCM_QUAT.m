function R = GetDCM_QUAT(q)

    q = q(:); qw=q(1); qv=q(2:4);

    Xi  = [-qv';  qw*eye(3)+[  0, -qv(3),  qv(2);
                                 qv(3), 0, -qv(1);
                                -qv(2), qv(1), 0 ]];
    
    Psi = [-qv';  qw*eye(3)-[  0, -qv(3),  qv(2);
                                 qv(3), 0, -qv(1);
                                -qv(2), qv(1), 0 ]];

    R = Psi' * Xi;

end
