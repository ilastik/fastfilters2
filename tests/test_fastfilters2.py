import fastfilters
import numpy
import pytest

import fastfilters2

# Kernel values are extracted from the original fastfilters library.
KERNELS_TEXT = """
0.3 0 0x1.fc125ap-1 0x1.f6d37ap-9
0.3 1 0x0p+0 0x1.fffff8p-2 0x1.f04ep-25
0.3 2 -0x1.b297d2p-1 0x1.dc3f88p-3 0x1.88f01ep-3
0.7 0 0x1.23c2d4p-1 0x1.a4a892p-3 0x1.3b317p-7 0x1.eaf0acp-15
0.7 1 0x0p+0 0x1.ae4b2p-2 0x1.426996p-5 0x1.78a37cp-12
0.7 2 -0x1.21997cp+0 0x1.b38f4ep-2 0x1.19b9b4p-3 0x1.6367bap-9
1.0 0 0x1.98a0a4p-2 0x1.efb0bp-3 0x1.ba69eap-5 0x1.228634p-8
1.0 1 0x0p+0 0x1.ef97dcp-3 0x1.ba53c2p-4 0x1.b3b37cp-7 0x1.18aef2p-11
1.0 2 -0x1.98c6ep-2 0x1.0caf58p-17 0x1.4bf464p-3 0x1.22b3aep-5 0x1.0857dcp-9
1.6 0 0x1.fee3dap-3 0x1.a43f28p-3 0x1.d3ce16p-4 0x1.605a9ep-5 0x1.6726acp-7 0x1.ef673ep-10
1.6 1 0x0p+0 0x1.48654ap-4 0x1.6d8f34p-4 0x1.9d0346p-5 0x1.18a744p-6 0x1.e3e836p-9 0x1.0efcb4p-11
1.6 2 -0x1.8f1af8p-4 -0x1.901acap-5 0x1.9b274ap-6 0x1.5a3c98p-5 0x1.7044eep-6 0x1.a82146p-8 0x1.273694p-10 0x1.052f06p-13
3.5 0 0x1.d35556p-4 0x1.c0a47ep-4 0x1.8cefdcp-4 0x1.43a924p-4 0x1.e67286p-5 0x1.50e606p-5 0x1.ae1212p-6 0x1.f9f938p-7 0x1.124dfep-7 0x1.121abap-8 0x1.f8ddb6p-10 0x1.ac80bep-11
3.5 1 0x0p+0 0x1.253fp-7 0x1.037324p-6 0x1.3d54d8p-6 0x1.3df4dcp-6 0x1.134234p-6 0x1.a5a90ep-7 0x1.21612p-7 0x1.66965ep-8 0x1.931dc2p-9 0x1.9c7e8ep-10 0x1.811cf2p-11 0x1.48a03ap-12 0x1.00a5a4p-13
3.5 2 -0x1.3216dap-7 -0x1.0dda76p-7 -0x1.5e21f6p-8 -0x1.c19398p-10 0x1.86a3dcp-10 0x1.cba17p-9 0x1.1137f4p-8 0x1.f1675ep-9 0x1.7bc67ap-9 0x1.f84fa8p-10 0x1.2893b8p-10 0x1.385facp-11 0x1.28f0e2p-12 0x1.00aeaep-13 0x1.985f32p-15
5.0 0 0x1.476faap-4 0x1.40f3d8p-4 0x1.2e43p-4 0x1.117f6ap-4 0x1.db88f4p-5 0x1.8d334p-5 0x1.3ec2b8p-5 0x1.eb8fdp-6 0x1.6c286ep-6 0x1.033262p-6 0x1.628266p-7 0x1.d1dbc8p-8 0x1.2616cp-8 0x1.64bf72p-9 0x1.9fc9dp-10 0x1.d19932p-11
5.0 1 0x0p+0 0x1.9b6516p-9 0x1.836fe4p-8 0x1.06ecfap-7 0x1.30c4dp-7 0x1.3e3492p-7 0x1.32706p-7 0x1.13a908p-7 0x1.d2c674p-8 0x1.75c43p-8 0x1.1c013ep-8 0x1.9a87a6p-9 0x1.1ab88ap-9 0x1.73899ep-10 0x1.d255dcp-11 0x1.17c01ep-11 0x1.410bb8p-12 0x1.609b84p-13 0x1.72cca2p-14
5.0 2 -0x1.a43f58p-9 -0x1.8b718cp-9 -0x1.45d8cp-9 -0x1.c13864p-10 -0x1.b72982p-11 0x1.7874a6p-21 0x1.687732p-11 0x1.2f1446p-10 0x1.6cd17ap-10 0x1.74d9d2p-10 0x1.558118p-10 0x1.1f3dbcp-10 0x1.c1a0fap-11 0x1.4a1abp-11 0x1.c91984p-12 0x1.2b961cp-12 0x1.74d69ep-13 0x1.b9aa5ep-14 0x1.f3609ap-15 0x1.0e78dp-15 0x1.1a7ff8p-16
10.0 0 0x1.478f58p-5 0x1.45ed1ep-5 0x1.4112e6p-5 0x1.39258p-5 0x1.2e603cp-5 0x1.211216p-5 0x1.1199ep-5 0x1.0061f6p-5 0x1.dbb6f6p-6 0x1.b4f324p-6 0x1.8d59acp-6 0x1.65be86p-6 0x1.3ee18ep-6 0x1.19695cp-6 0x1.ebbf5ep-7 0x1.a95f62p-7 0x1.6c4ba8p-7 0x1.34e246p-7 0x1.034b76p-7 0x1.af008p-8 0x1.62a4b2p-8 0x1.20e8c4p-8 0x1.d208dap-9 0x1.7422dap-9 0x1.263334p-9 0x1.cc8b18p-10 0x1.64e1f4p-10 0x1.11cd7p-10 0x1.9ff20ap-11 0x1.38cc04p-11 0x1.d1c63cp-12
10.0 1 0x0p+0 0x1.a28f98p-12 0x1.9c5452p-11 0x1.2d9c96p-10 0x1.845132p-10 0x1.d009aep-10 0x1.0785ep-9 0x1.20186p-9 0x1.31760ap-9 0x1.3ba418p-9 0x1.3eed9ep-9 0x1.3bda5p-9 0x1.332292p-9 0x1.25a204p-9 0x1.144956p-9 0x1.0010ap-9 0x1.d3d5e4p-10 0x1.a5778cp-10 0x1.769d8ap-10 0x1.48a43p-10 0x1.1ca666p-10 0x1.e6f7b8p-11 0x1.9b7662p-11 0x1.577e72p-11 0x1.1b5cf2p-11 0x1.ce0fc6p-12 0x1.7461acp-12 0x1.28ae7ap-12 0x1.d3650cp-13 0x1.6c0a66p-13 0x1.1862ccp-13 0x1.ab22fcp-14 0x1.41c66ap-14 0x1.df8366p-15 0x1.61689p-15 0x1.01a71p-15
10.0 2 -0x1.a4a1ap-12 -0x1.9e58aep-12 -0x1.8bcdaep-12 -0x1.6deaa4p-12 -0x1.4623a4p-12 -0x1.165d96p-12 -0x1.c19c04p-13 -0x1.4fad4p-13 -0x1.b77efap-14 -0x1.a9a2ep-15 0x1.d61ba8p-24 0x1.82e9aep-15 0x1.68e7d2p-14 0x1.f3495cp-14 0x1.2f6b04p-13 0x1.55b86p-13 0x1.6d3784p-13 0x1.772a34p-13 0x1.7541d8p-13 0x1.6976c2p-13 0x1.55e15cp-13 0x1.3c9796p-13 0x1.1f908cp-13 0x1.008f1ap-13 0x1.c227bp-14 0x1.84abd8p-14 0x1.4a83c8p-14 0x1.14facap-14 0x1.c9b966p-15 0x1.7511e6p-15 0x1.2c0efap-15 0x1.dc783ep-16 0x1.759044p-16 0x1.215032p-16 0x1.bad25cp-17 0x1.4f024ap-17 0x1.f55176p-18 0x1.7330a8p-18 0x1.1030f8p-18 0x1.8bc5p-19 0x1.1db04p-19
"""

KERNELS = {
    (float(items[0]), int(items[1])): numpy.fromiter(
        map(float.fromhex, items[2:]), dtype=numpy.float32
    )
    for line in KERNELS_TEXT.splitlines()
    if (items := line.strip().split())
}

RNG = numpy.random.default_rng(seed=42)


def random_array(shape):
    return RNG.integers(0, 256, size=shape, dtype=numpy.uint8).astype(numpy.float32)


def idfn(val):
    if isinstance(val, tuple):
        return "x".join(map(str, val))


class TestGaussianKernel:
    @pytest.mark.parametrize("order", [0, 1, 2])
    @pytest.mark.parametrize("scale", [0.3, 0.7, 1.0, 1.6, 3.5, 5.0, 10.0])
    def test_result(self, scale, order):
        actual = fastfilters2.gaussian_kernel(scale, order)
        desired = KERNELS[scale, order]
        numpy.testing.assert_array_almost_equal_nulp(actual, desired)

    @pytest.mark.parametrize("scale, ok", [(-1e-5, False), (0, False), (1e-5, True)])
    def test_non_positive_scale(self, scale, ok):
        if ok:
            fastfilters2.gaussian_kernel(scale)
        else:
            with pytest.raises(ValueError):
                fastfilters2.gaussian_kernel(scale)

    @pytest.mark.parametrize("order", [-1, 3])
    def test_invalid_order(self, order):
        with pytest.raises(ValueError):
            fastfilters2.gaussian_kernel(1, order)


@pytest.mark.parametrize("scale", [0.3, 0.7, 1.0, 1.6, 3.5, 5.0, 10.0])
@pytest.mark.parametrize("shape", [(512, 512), (64, 64, 64)], ids=idfn)
class TestFilters:
    def test_gaussian_smoothing(self, shape, scale):
        data = random_array(shape)
        actual = fastfilters2.gaussian_smoothing(data, scale)
        desired = fastfilters.gaussianSmoothing(data, scale)
        numpy.testing.assert_array_almost_equal_nulp(actual, desired, nulp=4)


@pytest.mark.skip(reason="not a bottleneck")
@pytest.mark.parametrize("scale, order", [(0.3, 0), (10, 2)])
def bench_gaussian_kernel(benchmark, scale, order):
    benchmark(fastfilters2.gaussian_kernel, scale, order)


@pytest.mark.parametrize("scale", [0.3, 10.0])
@pytest.mark.parametrize("shape", [(512, 512), (64, 64, 64)], ids=idfn)
class BenchFilters:
    @pytest.mark.skip(reason="Temporarily disabled for faster benchmarking")
    def bench_fastfilters1_gaussian_smoothing(self, benchmark, shape, scale):
        data = random_array(shape)
        benchmark(fastfilters.gaussianSmoothing, data, scale)

    def bench_fastfilters2_gaussian_smoothing(self, benchmark, shape, scale):
        data = random_array(shape)
        benchmark(fastfilters2.gaussian_smoothing, data, scale)
