#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <cmath>
#include <array>

constexpr size_t STAR_NUM = 48;

template <typename T,
    typename TIter = decltype(std::begin(std::declval<T>())),
    typename = decltype(std::end(std::declval<T>()))>
    constexpr auto enumerate(T&& iterable)
{
    struct iterator
    {
        size_t i;
        TIter iter;
        bool operator != (const iterator& other) const { return iter != other.iter; }
        void operator ++ () { ++i; ++iter; }
        auto operator * () const { return std::tie(i, *iter); }
    };
    struct iterable_wrapper
    {
        T iterable;
        auto begin() { return iterator{ 0, std::begin(iterable) }; }
        auto end() { return iterator{ 0, std::end(iterable) }; }
    };
    return iterable_wrapper{ std::forward<T>(iterable) };
}

float frand() {
    return (float)rand() / RAND_MAX * 2 - 1;
}

struct Star {
    float px, py, pz;
    float vx, vy, vz;
    float mass;
};

std::vector<Star> stars;

void init() {
    for (int i = 0; i < 48; i++) {
        stars.push_back({
            frand(), frand(), frand(),
            frand(), frand(), frand(),
            frand() + 1,
        });
    }
}

float G = 0.001;
float eps = 0.001;
float dt = 0.01;

void step() {
    std::array<float, STAR_NUM> delta_vx = { 0.f };
    std::array<float, STAR_NUM> delta_vy = { 0.f };
    std::array<float, STAR_NUM> delta_vz = { 0.f };

    for (auto& other : stars) {
        for (auto&& [i, star] : enumerate(stars)) {
            float dx = other.px - star.px;
            float dy = other.py - star.py;
            float dz = other.pz - star.pz;
            float d2 = dx * dx + dy * dy + dz * dz + eps * eps;
            d2 *= sqrt(d2);
            delta_vx[i] += dx * other.mass / d2;
            delta_vy[i] += dy * other.mass / d2;
            delta_vz[i] += dz * other.mass / d2;
        }
    }
    for (auto&& [i, star] : enumerate(stars)) {
        star.vx += delta_vx[i] * G * dt ;
        star.vy += delta_vy[i] * G * dt ;
        star.vz += delta_vz[i] * G * dt ;
        star.px += star.vx * dt;
        star.py += star.vy * dt;
        star.pz += star.vz * dt;
    }
}

float calc() {
    float energy = 0;
    for (auto&& [i, star] : enumerate(stars)) {
        float v2 = star.vx * star.vx + star.vy * star.vy + star.vz * star.vz;
        energy += star.mass * v2 / 2;
        for (auto&& [j, other] : enumerate(stars)) {
            if (i != j) {
                float dx = other.px - star.px;
                float dy = other.py - star.py;
                float dz = other.pz - star.pz;
                float d2 = dx * dx + dy * dy + dz * dz + eps * eps;
                energy -= other.mass * star.mass * G / sqrt(d2) * 0.5;
            }
        }
    }
    return energy;
}

template <class Func>
long benchmark(Func const &func) {
    auto t0 = std::chrono::steady_clock::now();
    func();
    auto t1 = std::chrono::steady_clock::now();
    auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
    return dt.count();
}

int main() {
    init();
    printf("Initial energy: %f\n", calc());
    auto dt = benchmark([&] {
        for (int i = 0; i < 100000; i++)
            step();
    });
    printf("Final energy: %f\n", calc());
    printf("Time elapsed: %ld ms\n", dt);
    return 0;
}
