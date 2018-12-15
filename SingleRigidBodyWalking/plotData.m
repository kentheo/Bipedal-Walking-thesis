% endpoint = size(store_grf,1);

figure(2); clf; 
subplot(3,1,1); hold on; title('contact forces')
plot(store_time,store_grf(:,1))
subplot(3,1,2); plot(store_time,store_grf(:,2))
subplot(3,1,3); plot(store_time,store_grf(:,3)/(uLINK(1).m*G) )

figure(3); clf; 
subplot(3,1,1); hold on; title('contact point position')
plot(store_time,store_p(:,1))
subplot(3,1,2); plot(store_time,store_p(:,2))
subplot(3,1,3);  %hold on; grid on; 
plot(store_time,store_p(:,3)) 

figure(4); clf; 
subplot(3,1,1); hold on;  title('contact point velocity')
plot(store_time,store_v(:,1))
plot(store_time,[0;diff(store_p(:,1))/logdata_skip],'r')
legend('store_v','diff(store_p)');
subplot(3,1,2); plot(store_time,store_v(:,2))
subplot(3,1,3); hold on; %grid on;
plot(store_time,store_v(:,3))
plot(store_time,store_v(:,3),'o')
plot(store_time,[0;diff(store_p(:,3))/logdata_skip],'r.')
legend('store_v','store_v','diff(store_p)');

% return;

figure(5); clf;
subplot(3,1,1); hold on; title('COM positiom')
plot(store_time,store_pcom(:,1))
% plot(time,store_vc(:,1),'g')
subplot(3,1,2); plot(store_time,store_pcom(:,2))
subplot(3,1,3); plot(store_time,store_pcom(:,3))

figure(6); clf; 
subplot(3,1,1); hold on; title('COM velocity')
plot(store_time,store_vcom(:,1))
% plot(time,store_vc(:,1),'g');
% plot(time,[0;diff(store_pcom(:,1))/Dtime],'g')
% legend('vcom')
subplot(3,1,2); hold on;
plot(store_time,store_vcom(:,2))
% plot(time,[0;diff(store_pcom(:,2))/Dtime],'g')
subplot(3,1,3); hold on;
plot(store_time,store_vcom(:,3))
% plot(time,store_vc(:,3),'g');
% plot(time,[0;diff(store_pcom(:,3))/Dtime],'g')
% legend('vcom','vcom')
%%
return
figure(7); clf; hold on;
plot(store_time,store_dleg(:,3))
plot(store_time,store_dleg(:,3),'o')
plot(store_time,[0; diff(store_leg(:,3))/logdata_skip] , 'r.')
legend('dleg','dleg','diff(leg)');