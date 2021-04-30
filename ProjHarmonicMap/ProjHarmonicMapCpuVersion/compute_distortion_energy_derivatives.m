function [Ed_g, Ed_h, e] = compute_distortion_energy_derivatives(Jp, hess_sample_rate, energy_type, z)
    J = z' * Jp;
    [u,s,v] = svdJacobian(J, 'avx');
    s = reshape(s,3,[]);
    Ed_g = compute_distortion_energy_gradient(u, s, v, energy_type, Jp) / size(Jp,2) * 3 ;
    Ed_h = compute_distortion_energy_hessian(u, s, v, energy_type, hess_sample_rate, Jp) / size(Jp,2) * 3;
    
    switch energy_type
        case 1
            s = s(:);
            e = sum(s.^2 + s.^(-2) - 2) / size(Jp,2) * 3;
        case 2
            s = s(:);
            e = sum((s-1).^2) / size(Jp,2) * 3;
    end
end

function Ed_g = compute_distortion_energy_gradient(u, s, v, energy_type, Jp)
    nj = size(Jp, 2)/3;
    g_diagscales = zeros(9, nj);
    
    v = reshape(v, 9, []);
    u = reshape(u, 9, []);
    v11 = v(1,:);  v12 = v(4,:);  v13 = v(7,:);
    v21 = v(2,:);  v22 = v(5,:);  v23 = v(8,:);
    v31 = v(3,:);  v32 = v(6,:);  v33 = v(9,:);
    
    u11 = u(1,:);  u12 = u(4,:);  u13 = u(7,:);
    u21 = u(2,:);  u22 = u(5,:);  u23 = u(8,:);
    u31 = u(3,:);  u32 = u(6,:);  u33 = u(9,:);
    
    s1 = s(1,:);
    s2 = s(2,:);
    s3 = s(3,:);
    switch energy_type
        case 1
            partial_s1 = 2*(s1 - s1.^(-3));
            partial_s2 = 2*(s2 - s2.^(-3));
            partial_s3 = 2*(s3 - s3.^(-3));
        case 2
            partial_s1 = 2*(s1 - 1);
            partial_s2 = 2*(s2 - 1);
            partial_s3 = 2*(s3 - 1);
        case 3
            
        otherwise
            error('Unknow energy type');
    end
    
    up11 = u11.*partial_s1; up12 = u12.*partial_s2; up13 = u13.*partial_s3;
    up21 = u21.*partial_s1; up22 = u22.*partial_s2; up23 = u23.*partial_s3;
    up31 = u31.*partial_s1; up32 = u32.*partial_s2; up33 = u33.*partial_s3;
    
    g_diagscales(1,:) = v11.*up11 + v12.*up12 + v13.*up13;
    g_diagscales(2,:) = v21.*up11 + v22.*up12 + v23.*up13;
    g_diagscales(3,:) = v31.*up11 + v32.*up12 + v33.*up13;
    g_diagscales(4,:) = v11.*up21 + v12.*up22 + v13.*up23;
    g_diagscales(5,:) = v21.*up21 + v22.*up22 + v23.*up23;
    g_diagscales(6,:) = v31.*up21 + v32.*up22 + v33.*up23;
    g_diagscales(7,:) = v11.*up31 + v12.*up32 + v13.*up33;
    g_diagscales(8,:) = v21.*up31 + v22.*up32 + v23.*up33;
    g_diagscales(9,:) = v31.*up31 + v32.*up32 + v33.*up33;
    
    g_diagscales = (reshape(g_diagscales,3,[]))';
    Ed_g = Jp*g_diagscales;
    Ed_g = Ed_g(:);
end

function Ed_h = compute_distortion_energy_hessian(u, s, v, energy_type, hess_sample_rate, Jp)
    nj = size(u,2)/3;
    idx = 1:hess_sample_rate:nj;    
    nds = length(idx);
    nvar = size(Jp,1);
    s1 = s(1,idx);
    s2 = s(2,idx);
    s3 = s(3,idx);
    h_diagscales = zeros(9,nds);
    switch energy_type 
        case 1
            h_diagscales(1:3,:) = 2 + 6 * s(:,idx).^(-4);
            
            a = 1 + (s1.*s2).^(-2);
            b = (s1.^2 + s2.^2).*(s1.*s2).^(-3);
            h_diagscales(4,:) = a + b;
            h_diagscales(5,:) = a - b;
            h_diagscales(5,find(h_diagscales(5,:)<0)) = 0;
            
            a = 1 + (s1.*s3).^(-2);
            b = (s1.^2 + s3.^2).*(s1.*s3).^(-3);
            h_diagscales(6,:) = a + b;
            h_diagscales(7,:) = a - b;
            h_diagscales(7,find(h_diagscales(7,:)<0)) = 0;
            
            a = 1 + (s2.*s3).^(-2);
            b = (s2.^2 + s3.^2).*(s2.*s3).^(-3);
            h_diagscales(8,:) = a + b;
            h_diagscales(9,:) = a - b;
            h_diagscales(9,find(h_diagscales(9,:)<0)) = 0;
        case 2
            h_diagscales(1:3,:) = 2;
            
            a = 1 - 1./(s1+s2);
            b = 1./(s1+s2);
            h_diagscales(4,:) = a + b;
            h_diagscales(5,:) = a - b;
            h_diagscales(5,find(h_diagscales(5,:)<0)) = 0;
            
            a = 1 - 1./(s1+s3);
            b = 1./(s1+s3);
            h_diagscales(6,:) = a + b;
            h_diagscales(7,:) = a - b;
            h_diagscales(7,find(h_diagscales(7,:)<0)) = 0;
            
            a = 1 - 1./(s2+s3);
            b = 1./(s2+s3);
            h_diagscales(8,:) = a + b;
            h_diagscales(9,:) = a - b;
            h_diagscales(9,find(h_diagscales(9,:)<0)) = 0;
    end
    h_diagscales = h_diagscales(:);
    M = zeros(9,9); 
    h_temp = zeros(3*nvar, 9*nds);
    n = 1;
    for ii = idx
        vi = v(:,3*ii-2:3*ii);
        ui = u(:,3*ii-2:3*ii);
        temp = kron(vi,ui);
        j = 9*(n-1);
        M(:,1) = temp(:,1);
        M(:,2) = temp(:,5);
        M(:,3) = temp(:,9);
        M(:,4) = temp(:,2) + temp(:,4);
        M(:,5) = temp(:,2) - temp(:,4);
        M(:,6) = temp(:,3) + temp(:,7);
        M(:,7) = temp(:,3) - temp(:,7);
        M(:,8) = temp(:,6) + temp(:,8);
        M(:,9) = temp(:,6) - temp(:,8);
        D = Jp(:,3*ii-2:3*ii);
        D = kron(eye(3),D);
        h_temp(:,j+1:j+9) = D*M;    
        n = n+1;
    end
    Ed_h = h_temp.*h_diagscales'*h_temp';
    Ed_h = Ed_h/nds*nj;
end