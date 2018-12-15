function [acc_linear,acc_angular] = NewtonEuler(j,ftot,tautot,dt)
global uLINK

acc_linear=ftot./uLINK(j).m;
uLINK(j).pcom = uLINK(j).pcom + uLINK(j).vcom*dt + 0.5*acc_linear*dt*dt;
uLINK(j).vcom = uLINK(j).vcom + acc_linear*dt;


uLINK(j).I = uLINK(1).R*uLINK(j).I_body*uLINK(1).R';
acc_angular = uLINK(j).I \ ( tautot- cross(uLINK(1).w,uLINK(j).I*uLINK(j).w) );
uLINK(j).R = Rodrigues(uLINK(j).w,dt) * uLINK(j).R ;
uLINK(j).w = uLINK(j).w + acc_angular*dt;



