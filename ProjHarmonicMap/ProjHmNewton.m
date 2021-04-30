classdef ProjHmNewton < handle
    
    properties
        x
        t
        CageX
        CageT
        newCageCoef
        edgeInfo
        p2p_weight
        smooth_weight
        pre_numPoint_constraints
        numPoint_constraints
        original_var
        variable
        isometric_sample_points
        lipschitz_sample_points
        epsilon_iso %isometric sampling radius
        epsilon_lips %lipschitz sampling radius
        hess_sample_step_size
        Proj
        DOF
        Ds
        doPreprocess
        status
        solver_type % 0 for Lu, 1 for cholesky

        max_number_adaptive_sampling 
        hasgpu    
        gcc
        %% variables for cpu_deformer
        D
        Kappa
        Ngamma
        smooth_energy_h     
        Jp
        L1
        p2p_matrix
        M
        b
    end
    
    methods(Static)
        function hasgpu = gpuDeviceTest
            if gpuDeviceCount == 0
                hasgpu = false;
            else
                dev = gpuDevice;
                if dev.TotalMemory <  8e9 % gpu total memory > 8GB 
                    hasgpu = false;
                else
                    hasgpu = true;
                end
            end
       end
   end
    
    methods
        function I = ProjHmNewton(x, t, cage, sampling, para)
            I.x = x;
            I.t = t;
            I.smooth_weight = para.smooth_weight;     
            I.CageX = cage.x;
            I.CageT = cage.t;
            I.gcc = green_coords_computer;
            I.gcc.create_computer(I.CageX, I.CageT);
            
            I.isometric_sample_points = sampling.points;
            I.epsilon_iso = sampling.eps;                       
            I.pre_numPoint_constraints = 0;
            I.numPoint_constraints = 0;
            
            I.hess_sample_step_size = para.step_size;  %
            I.DOF = para.dof;
            
            I.doPreprocess = false;
            
            I.solver_type = int32(0); %lu_solver
            
            I.hasgpu = I.gpuDeviceTest;
            I.max_number_adaptive_sampling = 1e5;
            I.newCageCoef = I.CageX;
%             I.preprocess();
        end
        
        function setGPUdeformer(I)
            if I.gpuDeviceTest == 0 %if machine dont have gpu device, do not set gpu deformer 
                return
            end
            I.doPreprocess = false;      
            I.hasgpu = ~I.hasgpu;            
            I.preprocess();
        end
        
           
        function set_smooth_weight(I, w_smooth)
           
            I.smooth_weight = w_smooth;
            ProjHarmonicMap('set_smooth_weight', I.smooth_weight / size(I.edgeInfo, 1));
            I.pre_numPoint_constraints = 0;
        end
       
        function set_sampling_points(I, lsp, eps)              
            I.lipschitz_sample_points = lsp;
            I.epsilon_lips = eps;
            if ~I.hasgpu
                    [~, I.Ngamma, ~] = lip_kernel_matrix_c(I.CageX', int32(I.CageT'-1), int32(I.edgeInfo - 1)', I.lipschitz_sample_points', I.epsilon_lips);               
                    [H_phi, H_psi] = I.gcc.compute_green_coords_hessian(I.lipschitz_sample_points);
                    I.L1 = [H_phi, H_psi] * I.Proj;            
            else
                 ProjHarmonicMap('set_sampling_points',I.lipschitz_sample_points', I.epsilon_lips);
            end
        end
        
        function preprocess(I)
%             I.ResetPointConstraints;
%             I.Reset;
            if I.doPreprocess
                return;
            end
            fprintf("====================================\n");
            fprintf("wait for preprocess!\n");
            
            nv = size(I.CageX, 1);
            nt = size(I.CageT, 1);
            Half_edge_spmat = sparse(I.CageT(:,[1,2,3]), I.CageT(:,[2,3,1]), repmat((1:nt)',1,3), nv, nv);
            [Half_edge1, Half_edge2, FaceId1] = find(tril(Half_edge_spmat));
            [~,~,FaceId2] = find(tril(Half_edge_spmat'));
            I.edgeInfo = [Half_edge1, Half_edge2, FaceId1, FaceId2];
            

            
            if size(I.Kappa,2) ~= I.DOF
                    I.Kappa = lip_kernel_matrix_c(I.CageX', int32(I.CageT'-1), int32(I.edgeInfo - 1)');
                    [I.Proj, ~] = eig(I.Kappa'*I.Kappa);
                    I.Proj = I.Proj(:,1:I.DOF);
                    I.Kappa = I.Kappa * I.Proj;
            end
            
            if I.hasgpu
                ProjHarmonicMap('set_DOF', I.DOF);
                I.original_var = ProjHarmonicMap('set_model',(I.CageX)',int32(I.CageT - 1)', int32(I.edgeInfo - 1)', (I.x)', I.Proj);                
%                 I.variable = I.original_var;
                ProjHarmonicMap('set_deformer',(I.isometric_sample_points)', ...
                    struct('epsilon', I.epsilon_iso, 'w_p2p', I.p2p_weight, 'w_smooth', I.smooth_weight / size(I.edgeInfo, 1), ... 
                'hess_sample_step_size', I.hess_sample_step_size, 'energy_type', int32(1),  'ls_step_size', 1));
                ProjHarmonicMap('set_solver',struct('solver',0));
                fprintf('Enable gpu deformer!');
            else                
                I.smooth_energy_h = I.Kappa'*I.Kappa;               
                FN = faceNormal(triangulation(I.CageT,I.CageX));
                I.original_var = I.Proj' * [I.CageX;FN];
%                 I.variable = I.original_var;
                
                if size(I.Jp,2) ~= 3 * size(I.isometric_sample_points,1) || size(I.D,2) ~= I.DOF
                    [phi,psi] = I.gcc.compute_green_coords(I.x);
                    I.D = [phi,psi] * I.Proj;
                    [J_phi, J_psi] = I.gcc.compute_green_coords_gradient(I.isometric_sample_points);
                    I.Jp = I.Proj' * [J_phi, J_psi]';
                end
                
                fprintf('Enable cpu deformer!');
            end
            
            if size(I.variable,1) ~= I.DOF
                    I.variable = I.original_var;
            end
                            
            I.set_sampling_points(I.isometric_sample_points, I.epsilon_iso);

            I.doPreprocess = true;
        end
        
        function set_p2p_weight(I, w_p2p)
             I.p2p_weight = w_p2p; 
             if I.hasgpu
                ProjHarmonicMap('set_p2p_weight', I.p2p_weight);
             end
        end
        
        function [viewY, Deformation_Converged] = run(I, numIter, P2PVtxIds, p2pDsts)  
            if  ~I.doPreprocess
                I.preprocess;
            end
            I.numPoint_constraints = length(P2PVtxIds);
            if ~I.numPoint_constraints
                I.p2p_matrix = zeros(0, size(I.CageX,1) + size(I.CageT,1));
                p2pDsts = zeros(0, 3);
            end  
                       
            if I.pre_numPoint_constraints ~= I.numPoint_constraints      
                qi = I.x(P2PVtxIds,:);
                if I.hasgpu
                    [I.variable, viewY, I.status, I.newCageCoef] = ProjHarmonicMap('deform',numIter,(I.variable), p2pDsts',qi');
                else 
                    viewY = cpu_deform(I, numIter, p2pDsts, qi);                   
                end
                I.pre_numPoint_constraints = I.numPoint_constraints;
            else
                if I.hasgpu
                    [I.variable, viewY, I.status, I.newCageCoef] = ProjHarmonicMap('deform',numIter,(I.variable), p2pDsts');
                else
                    viewY = cpu_deform(I, numIter, p2pDsts); 
                end
            end
            E = sum(p2pDsts(:).^2)*I.p2p_weight;           
            
                                    
            fprintf('enenergy = %f\n', E + I.status(2));
            fprintf('max_sigma1(sampling) | max_sigma1(estimate)|min_sigma3(sampling) | min_sigma3(estimate)|\n');
            fprintf('       %f      |        %f     |       %f      |        %f     |\n',I.status(4),I.status(5),I.status(6),I.status(7));
    
            Deformation_Converged = (abs(I.status(1,end) * I.status(end,end)) / (norm(I.variable, 'fro'))<1e-6) && norm(viewY(P2PVtxIds,:) - p2pDsts, 'fro') < 1e-3;
            
            if Deformation_Converged
                fprintf('====================================\n');
                fprintf('convergenced!\n');
            end
        end
       
        function viewY = cpu_deform(I, numIter, p2pDsts, qi)
            if nargin > 3
                [PHI, PSI] = I.gcc.compute_green_coords(qi);
                I.p2p_matrix = [PHI, PSI];
                I.p2p_matrix = I.p2p_matrix * I.Proj;                
            end
            I.M = I.smooth_energy_h * I.smooth_weight / size(I.edgeInfo, 1) + I.p2p_matrix'*I.p2p_matrix * I.p2p_weight;
            I.b = I.p2p_weight * I.p2p_matrix' * p2pDsts;
            for ii = 1:numIter
                [Eq_g, Eq_h, Eq] = compute_quad_energy_derivatives(I.M,I.b,I.variable);
               
                
                [Ed_g, Ed_h, Ed] = compute_distortion_energy_derivatives(I.Jp, I.hess_sample_step_size, 1, I.variable);
                E_h = Eq_h + Ed_h;
                E_g = Eq_g + Ed_g;
                
                
                delta = E_h\E_g;

                
                E = Eq + Ed;
                I.status(2) = E;
                I.status = linesearch_for_energy_decrease(I.M, I.b, I.Jp, I.variable, delta, E_g, 1, I.status);
                I.status = proj_linesearch_for_locally_injective(I.Jp, I.L1, I.Ngamma, I.Kappa, I.epsilon_iso, I.epsilon_lips, ...
                    I.variable, delta, I.isometric_sample_points, I.status, I.max_number_adaptive_sampling, I.Proj, 1, I.gcc);
                I.variable = I.variable - I.status(1) * reshape(delta,[],3);
                I.newCageCoef = I.Proj * I.variable;
                viewY = I.D*I.variable;
            end
        end
        
        
        function Reset(I)
            I.variable = I.original_var;
            I.pre_numPoint_constraints = 0;
        end
        
        function ResetPointConstraints(I)
            I.pre_numPoint_constraints = 0;
        end
        
    end
end