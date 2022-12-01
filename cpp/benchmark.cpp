#include <benchmark/benchmark.h>
#include <fastfilters2.h>

#include <random>

namespace ff2 = fastfilters2;

void bm_filters(benchmark::State &state) {
  std::random_device random_device;
  std::default_random_engine random_engine(random_device());
  std::uniform_int_distribution<> dist(0, 255);

  ff2::int_t size[] = {512, 512};
  constexpr ff2::int_t ndim = 2;

  std::vector<ff2::val_t> vdst(7 * size[0] * size[1]);
  std::vector<ff2::val_t> vsrc(size[0] * size[1]);

  auto dst = vdst.data();
  auto src = vsrc.data();
  ff2::val_t scale = 10;

  for (auto _ : state) {
    ff2::compute_filters(dst, src, size, ndim, scale);
    benchmark::DoNotOptimize(dst);
  }
}

BENCHMARK(bm_filters)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
