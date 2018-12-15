    if mod(time(n),logdata_skip) <eps
        if n==1
        %     store_w(n,1:3)=uLINK(1).w';
            store_time(n,1)=time(n);
            store_grf(n,1:3)=fc';
            store_p(n,1:3)=uLINK(1).joint(1).foot.p';
            store_v(n,1:3)=uLINK(1).joint(1).foot.v';
            store_vcom(n,1:3)=uLINK(1).vcom';
            store_pcom(n,1:3)=uLINK(1).pcom'; 
            store_leg(n,1:3) = uLINK(1).joint(1).leg';
            store_dleg(n,1:3) = uLINK(1).joint(1).dleg';
        else
            %     store_w(n,1:3)=uLINK(1).w';
            store_time(end+1,1)=time(n);
            store_grf(end+1,1:3)=fc';
            store_p(end+1,1:3)=uLINK(1).joint(1).foot.p';
            store_v(end+1,1:3)=uLINK(1).joint(1).foot.v';
            store_vcom(end+1,1:3)=uLINK(1).vcom';
            store_pcom(end+1,1:3)=uLINK(1).pcom';    
            store_leg(end+1,1:3) = uLINK(1).joint(1).leg';
            store_dleg(end+1,1:3) = uLINK(1).joint(1).dleg';
        end
    end