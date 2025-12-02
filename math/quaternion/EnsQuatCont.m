function q_continuous = EnsQuatCont(q_current, q_previous)
    % 쿼터니언 연속성 보장 함수
    % q_current: 현재 쿼터니언 (4x1)
    % q_previous: 이전 쿼터니언 (4x1)
    % q_continuous: 연속성이 보장된 쿼터니언 (4x1)
    
    % 열벡터로 보장
    q_current = q_current(:);
    q_previous = q_previous(:);
    
    % 두 쿼터니언의 내적 계산
    dot_product = q_current' * q_previous;
    
    % 내적이 음수면 부호를 바꿔서 최단 경로 선택
    if dot_product < 0
        q_continuous = -q_current;
    else
        q_continuous = q_current;
    end
end