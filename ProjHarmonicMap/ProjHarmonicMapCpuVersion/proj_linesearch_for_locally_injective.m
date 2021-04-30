function [status]  = proj_linesearch_for_locally_injective(Jp, L, Ngamma, Kappa, epsilon, epsilon2, z, delta, isp, status, max_number_adaptive_sampling, Proj, ls_step_size, gcc)
    delta_norm = norm(delta,'fro');
    delta = reshape(delta, [], 3);
    r1 = epsilon*sqrt(3)/2;   
    r2 = epsilon2*sqrt(2)/2;  

    
    ls = status(1);
    [coef_Jp1, coef_Jp2] = compute_Jacobian_coef(Jp, z, delta);
    [coef_vLip1, coef_vLip2, coef_vLip3, coef_NH1, coef_NH2] = compute_coef3(L, Ngamma, Kappa, z, delta);

    
    while ls*delta_norm > 1e-6
        z_next = z - ls*delta;
        s3_tmp = 1;       
        lips = compute_lips_next_step3(coef_vLip1, coef_vLip2, coef_vLip3, coef_NH1, coef_NH2, ls, r2);

        Jf = coef_Jp1 - coef_Jp2 * ls;
        sig3_low_bound = 1e-4; % sig3 - Lp*r > 1e-4
        [bound, flip_list, s3_tmp] = compute_sigma_value_next_step(Jf', lips, r1, s3_tmp, sig3_low_bound);
        
        if(bound(4) > 0)
            status(1) = ls;
            status(4:7) = bound;
            return;
        elseif ls > ls_step_size || bound(3) < 0
            ls = ls/2;
        else
            r1_tmp = r1 / 2;
            r2_tmp = r2 / 2;
            eps_tmp = epsilon / 2;
            lips_tmp = lips;
            bd = zeros(1,4);
            isp_tmp = isp;

            while length(flip_list) * 8 < max_number_adaptive_sampling && bd(3) >= 0
                isp_tmp = isp_tmp(flip_list,:);
                isp_tmp = [
                    isp_tmp + eps_tmp / 2 * [-1, -1, -1];
                    isp_tmp + eps_tmp / 2 * [-1, -1,  1];
                    isp_tmp + eps_tmp / 2 * [-1,  1, -1];
                    isp_tmp + eps_tmp / 2 * [-1,  1,  1];
                    isp_tmp + eps_tmp / 2 * [ 1, -1, -1];
                    isp_tmp + eps_tmp / 2 * [ 1, -1,  1];
                    isp_tmp + eps_tmp / 2 * [ 1,  1, -1];
                    isp_tmp + eps_tmp / 2 * [ 1,  1,  1];
                    ];
                    lips_tmp = lips_tmp(flip_list,:);
                    lips_tmp = repmat(lips_tmp,8,1);              
                [J_PHI, J_PSI] = gcc.compute_green_coords_gradient(isp_tmp);
                J_tmp = [J_PHI, J_PSI] * (Proj * z_next);
                [bd, flip_list, s3_tmp] = compute_sigma_value_next_step(J_tmp, lips_tmp, r1_tmp, s3_tmp, sig3_low_bound); 
                r1_tmp = r1_tmp / 2;
                eps_tmp = eps_tmp / 2;
                r2_tmp = r2_tmp / 2;
                if bd(4) > sig3_low_bound
                    status(1) = ls;
                    bound(4) = min(s3_tmp, bd(4));
                    status(4:7) = bound;
                    return
                end
            end
            ls = ls/2;
            
        end
        
    end
    ls = 0;
    status(1) = ls;
    z_next = z - ls*delta;     
    lips = compute_lips_next_step3(coef_vLip1, coef_vLip2, coef_vLip3, coef_NH1, coef_NH2, ls, r2);
    [status(4:7), ~, ~] = compute_sigma_value_next_step(Jp' *z_next, lips, r1, 1);
end


function lips = compute_lips_next_step3(coef_vLip1, coef_vLip2, coef_vLip3, coef_NH1, coef_NH2, ls, r2)
    lips = sqrt(coef_vLip1 - coef_vLip2 * ls + coef_vLip3 * ls^2);
    lips = lips + r2 * (coef_NH1 + coef_NH2 * ls);
    
end

function [coef_Jp1, coef_Jp2] = compute_Jacobian_coef(Jp, z, delta)
%% Jf(t) = coef_Jp1 - coef_Jp2*t
    coef_Jp1 = z' * Jp;
    coef_Jp2 = delta' * Jp;
end

function [coef_vLip1, coef_vLip2, coef_vLip3, coef_NH1, coef_NH2] = compute_coef3(L, Ngamma, Kappa, z, delta)  
    %% vLip = norm(L(z - t*delta)) = sqrt(coef_vLip1 - coef_vLip2 * t + coef_vLip3 * t^2)
    coef_vLip1 = transpose(sum(reshape(((L*z)'.^2),15,[])));
    coef_vLip3 = transpose(sum(reshape(((L*delta)'.^2),15,[])));
    coef_vLip2 = 2 * transpose(sum(reshape((L*z)'.*(L*delta)',15,[])));
    
    %% NH = |Ngamma| |K(z - t*delta)| <= |NGamma| |K*z| + |Ngamma| |K*delta| t
    ka_z = compute_kappa(Kappa, z);
    ka_d = compute_kappa(Kappa, delta);
    coef_NH1 = Ngamma * ka_z;
    coef_NH2 = Ngamma * ka_d;

end



function [bound, flip_list, s3_tmp] = compute_sigma_value_next_step(Jf, lips, r1, s3_tmp, sig3_low_bound)
    if nargin < 5
        sig3_low_bound = 0;
    end
    [~,s,~] = svdJacobian(Jf');
    s = reshape(s,3,[])';
    bound(1) = max(s(:,1));
    bound(2) = max(s(:,1) + lips*r1);
    bound(3) = min(s(:,3));
    bound(4) = min(s(:,3) - lips*r1);
    id = find(s(:,3) - lips*r1 > sig3_low_bound); 
    flip_list = find(s(:,3) - lips*r1 < sig3_low_bound);
    if isempty(id)       
        return;
    end
    if length(lips) > 1
        s3_tmp = min(min(s(id,3) - lips(id)*r1), s3_tmp);
    else
        s3_tmp = min(min(s(id,3) - lips*r1), s3_tmp);
    end
   
end

function ka = compute_kappa(Kappa, z)
    ka = Kappa * z;
    ka = (sum(reshape(ka', 9, []).^2).^0.5)';
end