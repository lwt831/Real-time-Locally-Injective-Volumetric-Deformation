function [Kappa, Ngamma, Gamma] = Lip_kernel_matrix(x,t,p_in,r)
nt = size(t,1);
nv = size(x,1);

Tri = triangulation(t,x);
FN = faceNormal(Tri);
Area2 = TriArea2(t,x);
Area2 = repmat(Area2,1,3);

Half_edge_spmat = sparse(t(:,[1,2,3]),t(:,[2,3,1]),repmat((1:nt)',1,3),nv,nv);
[Half_edge_triplet1_i,Half_edge_triplet1_j,Half_edge_triplet1_v] = find(tril(Half_edge_spmat));
[~,~,Half_edge_triplet2_v] = find(tril(Half_edge_spmat'));
edge = [Half_edge_triplet1_i,Half_edge_triplet1_j,Half_edge_triplet1_v,Half_edge_triplet2_v];

alpha = FN(edge(:,3),:);
tt = t(edge(:,3),:);
AA = Area2(edge(:,3),:);
dv1 = (x(tt(:,2),:) - x(tt(:,3),:))./AA;
dv2 = (x(tt(:,3),:) - x(tt(:,1),:))./AA;
dv3 = (x(tt(:,1),:) - x(tt(:,2),:))./AA;
alpha2Xdv1 = -cross(alpha,dv1);
alpha2Xdv2 = -cross(alpha,dv2);
alpha2Xdv3 = -cross(alpha,dv3);



p_alpha = -FN(edge(:,4),:);
p_tt = t(edge(:,4),:);
p_AA = Area2(edge(:,4),:);
p_dv1 = (x(p_tt(:,2),:) - x(p_tt(:,3),:))./p_AA;
p_dv2 = (x(p_tt(:,3),:) - x(p_tt(:,1),:))./p_AA;
p_dv3 = (x(p_tt(:,1),:) - x(p_tt(:,2),:))./p_AA;
p_alpha2Xdv1 = -cross(p_alpha,p_dv1);
p_alpha2Xdv2 = -cross(p_alpha,p_dv2);
p_alpha2Xdv3 = -cross(p_alpha,p_dv3);

Kappa =zeros(nv+nt,3*size(edge,1));
for ii = 1:size(edge,1)
    fid1 = edge(ii,3);
    fid2 = edge(ii,4);
    Kappa(nv+fid1,3*ii-2:3*ii) = alpha(ii,:);
    Kappa(nv+fid2,3*ii-2:3*ii) = p_alpha(ii,:);

    Kappa(t(fid1,1),3*ii-2:3*ii) =  alpha2Xdv1(ii,:);
    Kappa(t(fid1,2),3*ii-2:3*ii) =  alpha2Xdv2(ii,:);
    Kappa(t(fid1,3),3*ii-2:3*ii) =  alpha2Xdv3(ii,:);
    

    Kappa(t(fid2,1),3*ii-2:3*ii) =  Kappa(t(fid2,1),3*ii-2:3*ii) + p_alpha2Xdv1(ii,:);
    Kappa(t(fid2,2),3*ii-2:3*ii) =  Kappa(t(fid2,2),3*ii-2:3*ii) + p_alpha2Xdv2(ii,:);
    Kappa(t(fid2,3),3*ii-2:3*ii) =  Kappa(t(fid2,3),3*ii-2:3*ii) + p_alpha2Xdv3(ii,:);
end
Kappa = Kappa';

if nargout == 1
    return
end
    
Gamma = zeros(size(p_in,1), size(edge,1));
Ngamma = zeros(size(p_in,1), size(edge,1));

d = x(edge(:,1),:) - x(edge(:,2),:);
chunk_size = 100;
curr = 0;
r = r * sqrt(3)/3;
while curr <= size(p_in,1)
    if curr+chunk_size <= size(p_in,1)
        zt = p_in(curr+1:curr+chunk_size,:);
    else
        zt = p_in(curr+1:end,:);
    end
    epx = zt(:,1) - x(:,1)';
    epy = zt(:,2) - x(:,2)';
    epz = zt(:,3) - x(:,3)';
    ep = sqrt(epx.^2+epy.^2+epz.^2);
    e1 = ep(:,edge(:,1));
    e2 = ep(:,edge(:,2));
 
    l = sqrt(sum(d.^2,2));
    l = repmat(l',size(epx, 1),1);
    
    Sx = epx(:,edge(:,1))/r;
    Sy = epy(:,edge(:,1))/r;
    Sz = epz(:,edge(:,1))/r;
    Lx = epx(:,edge(:,2))/r;
    Ly = epy(:,edge(:,2))/r;
    Lz = epz(:,edge(:,2))/r;
    px = repmat(zt(:,1),1,size(e1, 2));
    Nx = zeros(size(px));
    Ny = Nx;
    Nz = Nx;
    a = Sx.*Sx + Sy.*Sy + Sz.*Sz;
    b = Sx.*Lx + Sy.*Ly + Sz.*Lz;
    c = Lx.*Lx + Ly.*Ly + Lz.*Lz;
    A = 4.*c.*(a.*c-b.^2);
    B = -4.*(a.*c-b.^2);
    C = (a+2.*b+c-4.*a.*c);
    D = 2.*(a-b);
    E = a - 1;
    s = solve_quartic_equation(A, B, C, D, E);    

    for ii = 1:4
        s{ii}.x = (-2 .* c .* s{ii}.y.^2 + s{ii}.y + 1)./(2 .* b .* s{ii}.y + 1);
        Idx = find(s{ii}.x > 0 & s{ii}.x < 1 & s{ii}.y > 0 & s{ii}.y < 1);
        Nx(Idx) = (s{ii}.x(Idx) .* Sx(Idx) + s{ii}.y(Idx) .* Lx(Idx)) * r;
        Ny(Idx) = (s{ii}.x(Idx) .* Sy(Idx) + s{ii}.y(Idx) .* Ly(Idx)) * r;
        Nz(Idx) = (s{ii}.x(Idx) .* Sz(Idx) + s{ii}.y(Idx) .* Lz(Idx)) * r;
    end
    Idx = find(abs(A)<1e-6);
    Nx(Idx) = Sx(Idx)./sqrt(a(Idx))*r;
    Ny(Idx) = Sy(Idx)./sqrt(a(Idx))*r;
    Nz(Idx) = Sz(Idx)./sqrt(a(Idx))*r;
    e1x = Sx*r - Nx;
    e1y = Sy*r - Ny;
    e1z = Sz*r - Nz;
    e2x = Lx*r - Nx;
    e2y = Ly*r - Ny;
    e2z = Lz*r - Nz;
    Re1 = sqrt(e1x.^2+e1y.^2+e1z.^2);
    Re2 = sqrt(e2x.^2+e2y.^2+e2z.^2);
    R1 = Re1 + Re2;
    Gamma(curr + 1:curr + size(epx, 1),:) = sqrt(2)*1./2./pi./sqrt(R1.^2-l.^2)./sqrt((e1-r).*(e2-r)).*l;    %|Gamma| approx sqrt(2)*|Gamma_2|
    Ngamma(curr + 1:curr + size(epx, 1),:) = 1/2/pi*1./(R1.^2-l.^2).*sqrt(10*R1.^2.*l.^2./(e1-r)./(e1-r)./(e2-r)./(e2-r) + 4); % |Nabla_Gamma| = sqrt(|Ngamma1|^2 + |Ngamma2|^2)
    curr = curr+chunk_size;
end

    
end
function s = solve_quartic_equation(A,B,C,D,E)
    p = (8.*A.*C-3.*B.^2)./(8.*A.^2);
    q = (B.^3-4.*A.*B.*C+8.*A.^2.*D)./(8.*A.^3);
    delta0 = C.^2 - 3.*B.*D+12.*A.*E;
    delta1 = 2.*C.^3-9.*B.*C.*D+27.*B.^2.*E+27.*A.*D.^2-72.*A.*C.*E;
%     delta = (delta1.^2-4.*delta0.^3)/-27;
    phi = acos(delta1./(2*delta0.^(3/2)));
    P = 1/2*sqrt(-2/3*p+2./(3*A).*sqrt(delta0).*cos(phi/3));
    k1 = max(-4*P.^2-2.*p+q./P,0);
    k2 = max(-4*P.^2-2.*p-q./P,0);
    s{1}.y = -B./(4.*A)-P+1/2*sqrt(k1);
    s{2}.y = -B./(4.*A)-P-1/2*sqrt(k1);
    s{3}.y = -B./(4.*A)+P+1/2*sqrt(k2);
    s{4}.y = -B./(4.*A)+P-1/2*sqrt(k2);
end
