clear HmDeformer tetMeshDeformer triMeshDeformer VHMDeformer ProjHmDeformer ProjHarmonicMap
wait(gpuDevice);
ProjHmDeformer = ProjHmNewton(x, t, cage, sampling, para1);