#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <cmath>

const size_t STAR_NUM = 48;

float frand() {
    return (float)rand() / RAND_MAX * 2 - 1;
}


float px[STAR_NUM], py[STAR_NUM], pz[STAR_NUM];
float vx[STAR_NUM], vy[STAR_NUM], vz[STAR_NUM];
float mass[STAR_NUM];




//Star<STAR_NUM> stars;

void init() {
    for (int i = 0; i < STAR_NUM; i++) {
        px[i] = frand();
        py[i] = frand();
        pz[i] = frand();
        vx[i] = frand();
        vy[i] = frand();
        vz[i] = frand();
        mass[i] = frand() + 1;
    }
}

const float G = 0.001;
const float eps = 0.001;
const float dt = 0.01;

void step() {
    for (int i = 0; i < STAR_NUM; i++) {
        for (int j = 0; j < STAR_NUM; j++) {
            float dx = px[j] - px[i];
            float dy = py[j] - py[i];
            float dz = pz[j] - pz[i];
            float d2 = dx * dx + dy * dy + dz * dz + eps * eps;
            d2 *= sqrt(d2);
            vx[i] += dx * mass[j] * G * dt / d2;
            vy[i] += dy * mass[j] * G * dt / d2;
            vz[i] += dz * mass[j] * G * dt / d2;
        }
    }
    for (int i = 0; i < STAR_NUM; i++) {
        px[i] += vx[i] * dt;
        py[i] += vy[i] * dt;
        pz[i] += vz[i] * dt;
    }
}

float calc() {
    float energy = 0;
    for (int i = 0; i < STAR_NUM; i++) {
        float v2 = vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i];
        energy += mass[i] * v2 / 2;
        for (int j = 0; j < STAR_NUM; j++) {
            float dx = px[j] - px[i];
            float dy = py[j] - py[i];
            float dz = pz[j] - pz[i];
            float d2 = dx * dx + dy * dy + dz * dz + eps * eps;
            energy -= mass[j] * mass[i] * G / sqrt(d2) / 2;
        }
    }
    return energy;
}

template <class Func>
long benchmark(Func const &func) {
    auto t0 = std::chrono::high_resolution_clock::now();
    func();
    auto t1 = std::chrono::high_resolution_clock::now();
    auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
    return dt.count();
}

int main() {
    init();
    printf("Initial energy: %f\n", calc());  // Initial energy: -8.571527
    /*auto dt = benchmark([&] {
        for (int i = 0; i < 100000; i++)
            step();
    });*/
    for (int i = 0; i < 100000; i++)
        step();
    printf("Final energy: %f\n", calc());  // Final energy: -8.511734
    //printf("Time elapsed: %ld ms\n", dt);
    return 0;
}
