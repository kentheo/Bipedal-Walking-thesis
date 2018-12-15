function [f_tot,fc,t_tot] = TopForce(j)
global uLINK G us uc

fg = [0 0 -uLINK(j).m * G]';    %  torque caused by gravity
tg = cross(uLINK(j).com, fg);            % torque applied from gravity in world coordinate

if uLINK(j).p(3) < 0.0  % contact point penetrates ground
    Kf = 1.0E+4;        % vertical stiffness 5.0E+3 to 5.0E+4«[N/m]
    Dfz=8E+4;       % vertical damping 5.0E+5 [N/(m/s)]
    v=uLINK(j).v; % contact point velocity
    fc(3,1) = -Kf*uLINK(j).p(3)-abs(uLINK(j).p(3))*Dfz*v(3); % non linear spring damper for vertical GRF only
    cv=1.0;   % coefficiant for exp to transit
    vth=0.10;  % velocity threshold 0.1 to 0.15 gives realistic motion but force is noisy
    Bd=0.0;   % viscous coefficient
    % friction force model
    if abs(v(1))>vth
        fc(1,1) =- sign(v(1))*( uc*fc(3,1) +(us-uc)*fc(3,1)*exp(-cv*abs(v(1))) ) - Bd*v(1);
    else
        fc(1,1) = -v(1)/vth*( uc*fc(3,1) +(us-uc)*fc(3,1)*exp(-cv*abs(v(1))) );
%          if abs(fc(1,1))>us*fc(3)
%             fc(1,1)=-sign(v(1))*us*fc(3);
%         end
    end    
     if abs(v(2))>vth
        fc(2,1) =- sign(v(2))*( uc*fc(3,1) +(us-uc)*fc(3,1)*exp(-cv*abs(v(2))) ) - Bd*v(2);
    else
        fc(2,1) = -v(2)/vth*( uc*fc(3,1) +(us-uc)*fc(3,1)*exp(-cv*abs(v(2))) );
%         if abs(fc(2,1))>us*fc(3)
%             fc(2,1)=-sign(v(2))*us*fc(3);
%         end
     end
    if fc(3)<0 % GRF only pushes, not pull
        fc=[0;0;0];
    end
else % pz>=0 case if not penetrates
    fc=[0;0;0];
end

f_tot = fg + fc;
t_tot = tg + cross(uLINK(j).p, fc); % total torque in the world is the sum of gravity torque and the torque created by GRF at contact point
