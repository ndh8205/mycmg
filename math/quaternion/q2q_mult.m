function quat_mult = q2q_mult(q,r) % Quaternion multifulication

    % Ensure both quaternions are column vectors
    q = q(:);
    r = r(:);
    
    a = q(1); b = q(2); c = q(3); d = q(4);
    e = r(1); f = r(2); g = r(3); h = r(4);
    
    quat_mult = [
        a*e - b*f - c*g - d*h;
        a*f + b*e + c*h - d*g;
        a*g - b*h + c*e + d*f;
        a*h + b*g - c*f + d*e
    ];
    

end


