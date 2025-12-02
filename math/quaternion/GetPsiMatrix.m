function Psi = GetPsiMatrix(q)
    % PDF의 Ψ(q) 행렬 구현 (오른쪽 곱셈 행렬)
    % q: 4x1 쿼터니언 [qw; qx; qy; qz]
    % Psi: 4x3 행렬
    
    q = q(:); % 열벡터 보장
    qw = q(1); qx = q(2); qy = q(3); qz = q(4);
    
    Psi = [-qx, -qy, -qz;
            qw,  qz, -qy;
           -qz,  qw,  qx;
            qy, -qx,  qw];
end