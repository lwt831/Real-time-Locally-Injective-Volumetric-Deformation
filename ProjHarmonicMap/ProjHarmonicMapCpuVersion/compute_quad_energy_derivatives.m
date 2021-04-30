function [Eq_g, Eq_h, e] = compute_quad_energy_derivatives(M, b, z)
    Eq_h = kron(eye(3), 2 * M);
    Eq_g = reshape(2 * (M * z -b), [], 1);
    e = trace(z'*M*z)-2*dot(b(:),z(:));
end