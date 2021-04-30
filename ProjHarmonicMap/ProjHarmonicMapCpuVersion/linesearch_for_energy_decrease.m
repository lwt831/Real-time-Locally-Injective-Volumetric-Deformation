function status = linesearch_for_energy_decrease(M, b, Jp, z, delta, Eg, energy_type, status)
    delta_norm = norm(delta);
    dot_delta_g = dot(delta, Eg);
    delta = reshape(delta,[],3);
    E = status(2);
    E_next = E;
    ls = 1;
    ls_alpha = 0.2;
    ls_beta = 0.5;
    while ls * delta_norm > 1e-6
        z_next = z - ls*delta;
        Eq_next = trace(z_next'*M*z_next) - 2 * dot(z_next(:),b(:));
        J_next = z_next' * Jp;
        [~,s,~] = svdJacobian(J_next);
        
        
        switch energy_type
            case 1
                s = s(:);
                Ed_next = sum(s.^2 + s.^(-2) - 2) / size(Jp,2) * 3;
            case 2
                s = s(:);
                Ed_next = sum((s-1).^2) / size(Jp,2) * 3;
        end
        E_next = Eq_next + Ed_next;
        if E_next < E - ls * ls_alpha * dot_delta_g
            break;
        end
        ls = ls * ls_beta;
    end 
    status(1) = ls;
    status(3) = E_next;
    status(8) = delta_norm;
    if energy_type == 2
        status(4) = max(s);
        status(6) = min(s);        
    end
end