function [fc,fdis,ftot, tcontact, taudis, tautot]= contactForce(j,i,dt)  % j is body id, i is joint id
global uLINK G us uc ground
% this contact model is only for a point. a better estimation can be
% formulated by penetration volumn similar to floatging law. the contact
% surface can be modeled a cone or sphere.

% ground.p=uLINK(j).joint(i).foot.p; % contact point position
% Ffilter=100.0;
% Tfilter=1/Ffilter;
% ground.v=(Tfilter*ground.v+dt*uLINK(j).joint(i).foot.v)./(Tfilter+dt);
% v=uLINK(j).joint(i).foot.v; % contact point velocity
% p=ground.p;
% v=ground.v;

p=uLINK(j).joint(i).foot.p;
v=uLINK(j).joint(i).foot.v;


if p(3) < 0.0  % contact point penetrates ground
    Kf = 2.0E+4;        % vertical stiffness 5.0E+3 to 5.0E+4 [N/m]  stiffness 1.0E+5 and damping 8.0E+5 are good over damped system
    Dfz=5.0E+4;       % vertical damping 5.0E+5 [N/(m/s)]
%  Kf = 2.0E+4; Dfz=5.0E+4;  are good stable set for rigid body drop
%  without leg compliance control
    fc(3,1) = -Kf*p(3)-abs(p(3))*Dfz*v(3); % non linear spring damper for vertical GRF only
%     pth=0.01;
%     fc(3,1) = -Kf*p(3)-(p(3)/pth)^2*Dfz*v(3); % non linear spring damper for vertical GRF only
%     if v(3)<=0
%         fc(3,1) = -Kf*p(3)-abs(p(3))*Dfz*v(3); % non linear spring damper for vertical GRF only
%     else
%         fc(3,1) = -Kf*p(3)-p(3)*p(3)*Dfz*v(3); % non linear spring damper for vertical GRF only
%     end
%     fc(3,1) = -Kf*p(3)-Dfz*v(3); % linear spring damper for vertical GRF only
    cv=1.0;   % coefficiant for exp to transit
    vth=0.15;  % velocity threshold 0.1 to 0.15 gives realistic motion but force is noisy
    Bd=0.1*uc;   % viscous coefficient
    % friction force model
    if abs(v(1))>vth
        fc(1,1) =- sign(v(1))*( uc*fc(3,1) +(us-uc)*fc(3,1)*exp(-cv*abs(v(1))) ) - Bd*v(1);
    else
        fc(1,1) = -v(1)/vth*( uc*fc(3,1) +(us-uc)*fc(3,1)*exp(-cv*abs(v(1))) )- Bd*v(1);
%          if abs(fc(1,1))>us*fc(3)
%             fc(1,1)=-sign(v(1))*us*fc(3);
%         end
    end    
     if abs(v(2))>vth
        fc(2,1) =- sign(v(2))*( uc*fc(3,1) +(us-uc)*fc(3,1)*exp(-cv*abs(v(2))) ) - Bd*v(2);
    else
        fc(2,1) = -v(2)/vth*( uc*fc(3,1) +(us-uc)*fc(3,1)*exp(-cv*abs(v(2))) )- Bd*v(2);
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

fg = [0 0 -uLINK(j).m * G]';    %  gravity
tcontact = cross(p-uLINK(j).pcom, fc);            % torque applied from contact point around the COM

fdis=[0 0 0]';
taudis=[0 0 0]';

ftot = fg + fc + fdis;
tautot =tcontact + taudis; % total torque applied around the COM