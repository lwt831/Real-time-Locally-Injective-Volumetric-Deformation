if size(viewY,1) ~= size(viewX,1)
    normY = vertexNormal(triangulation(t,viewY));
    viewY(end+1:end+numSeamVertex,:) = viewY(seamVertexIndex,:);
    normY(end+1:end+numSeamVertex,:) = normY(seamVertexIndex,:);
end