from src  import dgps, methods, metrics
import pytest
import numpy as np

# test data_preparation.DistributionSampler in data_preparation
with pytest.raises(ValueError, match='Unknown distribution'):
    sampler = dgps.DistributionSampler('hello',[1,2])

with pytest.raises(ValueError, match='parameters'):
    dgps.DistributionSampler('bimodal',[1,2])

# test invert_symmetric_matrix
with pytest.raises(ValueError, match='square'):
    metrics.invert_symmetric_matrix(np.ones((1,2)))

# test kde in mulkde
with pytest.raises(TypeError, match='number'):
    methods.kde('a', np.ones(3))

# test adtive kde in other_kde
with pytest.raises(ValueError, match ='bandwidth'):
    methods.adaptive_kde(-1,np.array([1,2,3]))

print("✅ Function tests all passed!")