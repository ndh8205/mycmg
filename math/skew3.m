%% skew3 Function

function skew_X = skew3( omega )

    a = omega(1);
    b = omega(2);
    c = omega(3);

    skew_X = [ 0, -c,  b;
               c,  0,  -a;
              -b,  a,  0];
    
    
end