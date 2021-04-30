modeldir= dir('model');
ismodel = arrayfun(@(i) modeldir(i).name(1)~='.', 1:numel(modeldir));
models = {modeldir(ismodel).name};

addpath('model');