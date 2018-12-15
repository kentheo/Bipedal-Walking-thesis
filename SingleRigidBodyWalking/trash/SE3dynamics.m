function [P,L] = SE3dynamics(j,f,tau)
global uLINK
% p_c = uLINK(j).R * uLINK(j).c; % relative position vector from contact point to COM
% w_c =p_c + uLINK(j).p;   % center of mass in world coordinate
w_c =uLINK(1).pcom;   % center of mass in world coordinate
w_I = uLINK(j).R * uLINK(j).I * uLINK(j).R';  % inertia around local COM in world frame oritentation
c_hat = hat(w_c);
Iww = w_I + uLINK(j).m * c_hat * c_hat';    % inertia tensor in world frame, it considers the displacement of COM
Ivv = uLINK(j).m * eye(3);
Iwv = uLINK(j).m * c_hat;

P = uLINK(j).m * (uLINK(j).vo + cross(uLINK(j).w,w_c));     % original: linear momentum
% P = uLINK(j).m * uLINK(j).vo;     % linear momentum, vo is the velocity of COM
L = uLINK(j).m * cross(w_c,uLINK(j).vo) + Iww * uLINK(j).w; % original: angular momentum: caused by COM particle + rotation around COM
% w_vc=uLINK(j).vo + cross(uLINK(j).w,p_c);
% L = uLINK(j).m * cross(w_c,uLINK(j).vo) +  w_I * uLINK(j).w; % angular momentum: caused by COM particle + rotation around COM

pp = [cross(uLINK(j).w,P);
    cross(uLINK(j).vo,P) + cross(uLINK(j).w,L)];
if nargin == 3
    pp = pp - [f; tau];   % net force torque vector
end
Ia = [Ivv, Iwv'; Iwv, Iww]; % spatial inertia tensor matrix

a0 = -Ia \ pp; % spatial acceleration
uLINK(j).dvo = a0(1:3);
uLINK(j).dw  = a0(4:6);
