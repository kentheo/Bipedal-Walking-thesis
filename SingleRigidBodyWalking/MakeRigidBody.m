function MakeRigidBody(j, wdh, mass)
global uLINK

uLINK(j).m = mass;  % 
uLINK(j).joint(1).offset = [0 0 -0.5*wdh(3)]';   % vector from COM to joint in body frame
uLINK(j).I_body = [1/12*(wdh(2)^2 + wdh(3)^2) 0 0;
            0 1/12*(wdh(1)^2 + wdh(3)^2)  0;
            0 0 1/12*(wdh(1)^2 + wdh(2)^2)] * mass; % inertia tensor
vert = [0      0      0;
   0      wdh(2) 0;
   wdh(1) wdh(2) 0;
   wdh(1) 0      0;
   0      0      wdh(3);
   0      wdh(2) wdh(3);
   wdh(1) wdh(2) wdh(3);
   wdh(1) 0      wdh(3)]';
%
for n=1:3
    uLINK(j).vertex(n,:) = vert(n,:) - wdh(n)/2;%  +uLINK(j).c(n);  % make box edges
end
clear vert;
uLINK(1).face = [
   1 2 3 4; 2 6 7 3; 4 3 7 8;
   1 5 8 4; 1 2 6 5; 5 6 7 8;
]';                             % make face mesh
