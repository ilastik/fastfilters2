#include <random>

#include <benchmark/benchmark.h>

#include <fastfilters2.h>

namespace ff = fastfilters2;
using std::size_t;

void bm_filters(benchmark::State &state) {
    std::random_device random_device;
    std::default_random_engine random_engine(random_device());
    std::uniform_int_distribution<> dist(0, 255);

    ff::shape_type shape{512, 512, 1};
    auto size = shape[0] * shape[1] * shape[2];
    size_t ndim = 2;
    auto scale = 10.0;

    std::vector<float> data(size);
    for (auto &v : data) {
        v = dist(random_engine);
    }

    std::vector<float> output(7 * size);

    for (auto _ : state) {
        ff::filters f{data.data(), shape, ndim, scale};
        auto out = output.data();

        f.gaussian_smoothing(out);
        out += size;

        f.gaussian_gradient_magnitude(out);
        out += size;

        f.laplacian_of_gaussian(out);
        out += size;

        f.hessian_of_gaussian_eigenvalues(out);
        out += 2 * size;

        f.structure_tensor_eigenvalues(out);
        out += 2 * size;

        benchmark::DoNotOptimize(out);
    }
}

BENCHMARK(bm_filters)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
