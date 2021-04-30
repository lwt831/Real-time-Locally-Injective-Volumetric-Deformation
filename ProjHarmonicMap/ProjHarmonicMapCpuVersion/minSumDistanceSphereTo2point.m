function d = minSumDistanceSphereTo2point(p0, r, p1, p2) %p0 is center point of shpere
    S = p1 - p0;
    L = p2 - p0;
    S = S/r;
    L = L/r;
    a = dot(S,S);
    b = dot(S,L);
    c = dot(L,L);
    A = 4*c*(a*c-b^2);
    B = -4*(a*c-b^2);
    C = (a+2*b+c-4*a*c);
    D = 2*(a-b);
    E = a - 1;
%     A= 1;B=0;C=0;D=0;E=-1;
    x1 = -1;
    x2 = -1;
    x3 = -1;
    x4 = -1;
    y1 = -1;
    y2 = -1;
    y3 = -1;
    y4 = -1;
    N = p0;
    if abs(A)<1e-6
        N = S/norm(S)*r + N;
    else     
        p = (8*A*C-3*B^2)/(8*A^2);
        q = (B^3-4*A*B*C+8*A^2*D)/(8*A^3);
        delta0 = C^2 - 3*B*D+12*A*E;
        delta1 = 2*C^3-9*B*C*D+27*B^2*E+27*A*D^2-72*A*C*E;
        delta = (delta1^2-4*delta0^3)/-27;
        if(delta < 0)
            Q = ((delta1+sqrt(delta*-27))/2)^(1/3);
            S = 1/2*sqrt(-2/3*p+1/(3*A)*(Q+delta0/Q));
            
        else
            phi = acos(delta1/(2*delta0^(3/2)));
            S = 1/2*sqrt(-2/3*p+2/(3*A)*sqrt(delta0)*cos(phi/3));
        end
        k1 = -4*S^2-2*p+q/S;
        k2 = -4*S^2-2*p-q/S;
        if k1 > 0
            y1 = -B/(4*A)-S+1/2*sqrt(k1);
            y2 = -B/(4*A)-S-1/2*sqrt(k1);
            x1 = (-2*c*y1^2+y1+1)/(2*b*y1+1);
            x2 = (-2*c*y2^2+y2+1)/(2*b*y2+1);
        end
        if k2 > 0
            y3 = -B/(4*A)+S+1/2*sqrt(k2);
            x3 = (-2*c*y3^2+y3+1)/(2*b*y3+1);
            y4 = -B/(4*A)+S-1/2*sqrt(k2);
            x4 = (-2*c*y4^2+y4+1)/(2*b*y4+1);
        end
        if x1>0&&x1<1 && y1>0&&y1<1
            N = N + x1*S+y1*L;
        elseif x2>0&&x2<1 && y2>0&&y2<1
            N = N + x2*S+y2*L;
        elseif x3>0&&x3<1 && y3>0&&y3<1
            N = N + x3*S+y3*L;
        elseif x4>0&&x4<1 && y4>0&&y4<1
            N = N + x4*S+y4*L;
        end
    end
    d = norm(N-p1)+norm(N-p2);
end