% clear all;
close all;

cv=1.0;   % coefficiant for exp to transit
vth=0.01;  % velocity threshold
fz=1.0; % unit normal force
us=0.8;
uc=0.6;
Bd=0.1*uc;   % viscous coefficient

velocity=-5:0.001:5;
for i=1:length(velocity)
    v=velocity(i);
    if abs(v)>=vth
        fc(i) =-sign(v)*( uc*fz +(us-uc)*fz*exp(-cv*abs(v)) ) - Bd*v;
    else
        fc(i)=-v/vth*( uc*fz +(us-uc)*fz*exp(-cv*abs(v)) ) - Bd*v;
    end
end

figure;
plot(velocity,fc)