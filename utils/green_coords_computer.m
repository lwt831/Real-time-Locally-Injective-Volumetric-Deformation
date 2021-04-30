classdef green_coords_computer < handle
    properties(SetAccess = protected)
        handle = -1     
    end
    
    methods
        function gc_computer = green_coords_computer
            gc_computer.handle = -1;
        end
        
        function create_computer(I, x, t)
            I.handle = green_coords_3d_urago3_imp(0, x', int32(t') - 1);
        end
        
        function [phi, psi] = compute_green_coords(I, p_in) 
            [phi, psi] = green_coords_3d_urago3_imp(1, I.handle, p_in');
        end
        
        function [J_phi, J_psi] = compute_green_coords_gradient(I, p_in)
            [J_phi, J_psi] = green_coords_3d_urago3_imp(2, I.handle, p_in');
        end
        
        function [H_phi, H_psi] = compute_green_coords_hessian(I, p_in)
            [H_phi, H_psi] = green_coords_3d_urago3_imp(3, I.handle, p_in');
        end
        
        function delete(I)
            if I.handle ~= -1
                green_coords_3d_urago3_imp(-1, I.handle);
                fprintf('gc_computer clean up!\n');
            end
        end
    end
end

