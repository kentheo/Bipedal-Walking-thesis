if mod(time(n),animation_skip) <eps
        show_GRF=[[uLINK(1).joint(1).foot.p(1); uLINK(1).joint(1).foot.p(1)+0.2*fc(1)/uLINK(1).m],[uLINK(1).joint(1).foot.p(2) ;uLINK(1).joint(1).foot.p(2)+0.2*fc(2)/uLINK(1).m] , [uLINK(1).joint(1).foot.p(3) ;uLINK(1).joint(1).foot.p(3)+0.2*fc(3)/uLINK(1).m]];
        show_vcom = [ [uLINK(1).pcom(1);uLINK(1).pcom(1)+0.5*uLINK(1).vcom(1)], [uLINK(1).pcom(2);uLINK(1).pcom(2)+0.5*uLINK(1).vcom(2)], [uLINK(1).pcom(3);uLINK(1).pcom(3)+0.5*uLINK(1).vcom(3)] ];
         show_leg = [ [uLINK(1).joint(1).p(1);uLINK(1).joint(1).foot.p(1)], [uLINK(1).joint(1).p(2);uLINK(1).joint(1).foot.p(2)], [uLINK(1).joint(1).p(3);uLINK(1).joint(1).foot.p(3)] ];
        vert = uLINK(1).R * uLINK(1).vertex;
        for k = 1:3
            vert(k,:) = vert(k,:) + uLINK(1).pcom(k);   % adding x,y,z to all vertex
        end
        color = [0.9 0.9 0.9];
        hold off;
        newplot;
        hold on;    
        plot3(uLINK(1).joint(1).p(1),uLINK(1).joint(1).p(2), uLINK(1).joint(1).p(3),'r.','color',[0.5 0 0],'markersize',40);  
        plot3(uLINK(1).joint(1).foot.p(1),uLINK(1).joint(1).foot.p(2), uLINK(1).joint(1).foot.p(3),'g.','color',[0 .5 0],'markersize',20);     
        h = patch('faces',uLINK(1).face','vertices',vert','FaceColor',color);      
        % plot COM
        [x y z]=sphere(10);
        r=0.06;
        face=uLINK(1).pcom;
        surf(face(1)+r*x,face(2)+r*y,face(3)+r*z,'FaceColor',[0 .5 .0],'EdgeColor','none','LineStyle','none','FaceLighting','phong');
        plot3(show_GRF(:,1),show_GRF(:,2) , show_GRF(:,3),'r','linewidth',3);
        plot3(show_vcom(:,1),show_vcom(:,2) , show_vcom(:,3),'g','linewidth',1);
        plot3(show_vcom(:,1),show_vcom(:,2) , show_vcom(:,3),'g','linewidth',1);
        plot3(show_leg(:,1),show_leg(:,2) , show_leg(:,3),'color',[0 0 0.7],'linewidth',1);
        hold on;
        axis equal
%         view(90,90);
        view(0,0);
%         view(0,90);
%         view(3);
        xlim(AX); ylim(AY); zlim(AZ);
        grid on
        text(0.25, -0.25, 0.75, ['time=',num2str(time(n),'%5.3f')]);
        text(0.25, 0, 0.55, ['pz= ',num2str(1000*uLINK(1).joint(1).foot.p(3),'%5.2f')]); 
        text(0.25, 0, 0.45, ['vz= ',num2str(uLINK(1).joint(1).foot.v(3),'%3.5f'),'    vx= ',num2str(uLINK(1).joint(1).foot.v(1),'%3.5f') ]); 
        text(0.25,0, 0.35, ['fz=',num2str(fc(3),'%5.3f'), ' N'] );
        text(0.25,0, 0.25, ['fg=',num2str(fg,'%5.3f'), ' N'] );        
        drawnow;
        if n==1
            pause(1);
        else
            pause(0.001);
        end
end