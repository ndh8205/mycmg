function [angle, axis] = Quat2AngleAxis(q)
% Quat2AngleAxis - 쿼터니언을 등가 회전 각도와 회전 축으로 변환합니다.
%
%   [angle, axis] = Quat2AngleAxis(q)
%
%   입력:
%       q: 4x1 쿼터니언 [qw; qx; qy; qz], 여기서 qw는 스칼라 성분입니다.
%
%   출력:
%       angle: 회전 각도 (라디안 단위, 0 ~ pi).
%       axis:  회전 축을 나타내는 3x1 단위 벡터.
%

% 입력 쿼터니언을 정규화하여 단위 쿼터니언으로 만듭니다.
q = q / norm(q);

% 부동 소수점 오차를 방지하기 위해 스칼라 성분을 [-1, 1] 범위로 제한합니다.
qw = min(max(q(1), -1.0), 1.0);

% 회전 각도를 계산합니다.
angle = 2 * acos(qw);

% 특이점(singularity) 경우를 확인합니다 (회전이 거의 없는 경우).
if abs(sin(angle/2)) < 1e-12
    % 회전이 없으면 각도는 0이며, 축은 임의로 설정할 수 있습니다 (예: x-축).
    angle = 0;
    axis = [1; 0; 0];
else
    % 회전 축을 계산합니다.
    axis = q(2:4) / sin(angle/2);
    % 안전을 위해 축 벡터를 다시 정규화합니다.
    axis = axis / norm(axis);
end

end