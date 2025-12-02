% inv_q.m
function inv_q = inv_q(q)
    conj_q = Quaternion_Conj(q);
    inv_q = conj_q(:) / norm(q)^2; 
end