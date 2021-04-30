clearvars -except model_name
load(model_name);
nViewV = size(viewX,1);
normX = vertexNormal(triangulation(t,x));
numSeamVertex = size(seamVertexIndex,1);
normX(end+1 : end+numSeamVertex, :) = normX(seamVertexIndex, :);
viewY = viewX;
normY = normX;

%% init parameters
Deformer_Method = {'ProjHmDeformer'};
Default_Deformer_Method = 'ProjHmDeformer';
Deformation_Converged = 1;
p2p_weight = 1e3;