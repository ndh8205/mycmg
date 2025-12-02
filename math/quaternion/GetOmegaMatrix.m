function Omega = GetOmegaMatrix(omega)
    % PDF의 Ω(ω) 행렬 구현
    % omega: 3x1 각속도 벡터
    % Omega: 4x4 skew-symmetric 행렬
    
    omega = omega(:); % 열벡터 보장
    
    Omega = [0,        -omega(1), -omega(2), -omega(3);
             omega(1),  0,         omega(3), -omega(2);
             omega(2), -omega(3),  0,         omega(1);
             omega(3),  omega(2), -omega(1),  0];
end