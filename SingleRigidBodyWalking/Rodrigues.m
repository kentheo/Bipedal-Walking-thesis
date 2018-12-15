function R = Rodrigues(w,dt)
% w should be column vector of size 3
% you can either specify w and dt or directly use theta=w*dt to get
% rotational angle
if norm(w)<eps
    wn=[0 0 0]';
else    
    wn = w/norm(w);		% normarized vector
end    
theta = norm(w)*dt;  % rotational angle given w and delta t
w_wedge = [0 -wn(3) wn(2);wn(3) 0 -wn(1);-wn(2) wn(1) 0];
R = eye(3) + w_wedge * sin(theta) + w_wedge^2 * (1-cos(theta));

% the rotational operation around the vector w