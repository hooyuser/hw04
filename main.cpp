#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <immintrin.h>

const size_t STAR_NUM = 48;

const float G = 0.001;
const float eps = 0.001;
const float dt = 0.01;
const float G_dt = G * dt;

alignas(32) float px[STAR_NUM];
alignas(32) float py[STAR_NUM];
alignas(32) float pz[STAR_NUM];
alignas(32) float vx[STAR_NUM];
alignas(32) float vy[STAR_NUM];
alignas(32) float vz[STAR_NUM];
alignas(32) float mass[STAR_NUM];

float frand() {
	return (float)rand() / RAND_MAX * 2 - 1;
}

static float reduce256_add_ps(__m256 ymm) {  // ymm = ( x7, x6, x5, x4, x3, x2, x1, x0 )

	__m128 vlow = _mm256_castps256_ps128(ymm);
	// vlow = ( x3, x2, x1, x0 )

	__m128 vhigh = _mm256_extractf128_ps(ymm, 1);
	// vhigh = ( x7, x6, x5, x4 )

	vlow = _mm_add_ps(vlow, vhigh);
	// vlow = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )

	__m128 shuf = _mm_movehdup_ps(vlow);
	// shuf = ( x3 + x7, x3 + x7, x1 + x5, x1 + x5 )

	__m128 sums = _mm_add_ps(vlow, shuf);
	// sum = ( - , x2 + x6 + x3 + x7, - , x0 + x4 + x1 + x5)

	shuf = _mm_movehl_ps(shuf, sums); // result = (shuf high 64 bits, sums high 64 bits)
	// shuf = ( x3 + x7, x3 + x7, - , x2 + x6 + x3 + x7 )

	sums = _mm_add_ss(sums, shuf);
	// sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )

	return _mm_cvtss_f32(sums);
}

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

void step() {
	const alignas(32) __m256 eps2 = _mm256_set1_ps(eps * eps);
	const alignas(32) __m256 dt_vec = _mm256_set1_ps(dt);

	for (size_t i = 0; i < STAR_NUM; i++) {
		__m256 px_i = _mm256_broadcast_ss(&px[i]);
		__m256 py_i = _mm256_broadcast_ss(&py[i]);
		__m256 pz_i = _mm256_broadcast_ss(&pz[i]);

		__m256 d_vx = _mm256_setzero_ps();
		__m256 d_vy = _mm256_setzero_ps();
		__m256 d_vz = _mm256_setzero_ps();

#pragma unroll 6
		for (size_t j = 0; j < STAR_NUM; j += 8) {
			__m256 px_j = _mm256_load_ps(&px[j]);
			__m256 dx = _mm256_sub_ps(px_j, px_i);

			__m256 py_j = _mm256_load_ps(&py[j]);
			__m256 dy = _mm256_sub_ps(py_j, py_i);

			__m256 pz_j = _mm256_load_ps(&pz[j]);
			__m256 dz = _mm256_sub_ps(pz_j, pz_i);
			//float dx = px[j] - px[i];
			//float dy = py[j] - py[i];
			//float dz = pz[j] - pz[i];

			__m256 prod = _mm256_mul_ps(dx, dx);
			prod = _mm256_fmadd_ps(dy, dy, prod);
			prod = _mm256_fmadd_ps(dz, dz, prod);
			prod = _mm256_add_ps(eps2, prod);
			//float prod = dx * dx + dy * dy + dz * dz + eps * eps;

			__m256 inverse_sqrt = _mm256_rsqrt_ps(prod);
			__m256 inverse_sqrt2 = _mm256_mul_ps(inverse_sqrt, inverse_sqrt);
			inverse_sqrt = _mm256_mul_ps(inverse_sqrt2, inverse_sqrt);
			//float inverse_sqrt = 1 / sqrt(prod)^3;

			__m256 mass_j = _mm256_load_ps(&mass[j]);
			__m256 factor = _mm256_mul_ps(inverse_sqrt, mass_j);
			//float factor = mass[j] / sqrt(prod)^3;

			d_vx = _mm256_fmadd_ps(dx, factor, d_vx);
			d_vy = _mm256_fmadd_ps(dy, factor, d_vy);
			d_vz = _mm256_fmadd_ps(dz, factor, d_vz);
			//d_vx += dx * factor;
			//d_vy += dy * factor; 
			//d_vz += dz * factor; 
		}
		vx[i] += reduce256_add_ps(d_vx) * G_dt;
		vy[i] += reduce256_add_ps(d_vy) * G_dt;
		vz[i] += reduce256_add_ps(d_vz) * G_dt;
	}
	//for (size_t i = 0; i < STAR_NUM; i++) {
	//	px[i] += vx[i] * dt;
	//	py[i] += vy[i] * dt;
	//	pz[i] += vz[i] * dt;
	//}
	for (size_t i = 0; i < STAR_NUM; i += 8) {
		__m256 px_i = _mm256_load_ps(&px[i]);
		__m256 vx_i = _mm256_load_ps(&vx[i]);
		_mm256_store_ps(&px[i], _mm256_fmadd_ps(vx_i, dt_vec, px_i));

		__m256 py_i = _mm256_load_ps(&py[i]);
		__m256 vy_i = _mm256_load_ps(&vy[i]);
		_mm256_store_ps(&py[i], _mm256_fmadd_ps(vy_i, dt_vec, py_i));

		__m256 pz_i = _mm256_load_ps(&pz[i]);
		__m256 vz_i = _mm256_load_ps(&vz[i]);
		_mm256_store_ps(&pz[i], _mm256_fmadd_ps(vz_i, dt_vec, pz_i));
		//px[i] += vx[i] * dt;
		//py[i] += vy[i] * dt;
		//pz[i] += vz[i] * dt;
	}
}

float calc() {
	float energy = 0;
	for (size_t i = 0; i < STAR_NUM; i++) {
		float v2 = vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i];
		energy += mass[i] * v2 / 2;
		for (size_t j = 0; j < STAR_NUM; j++) {
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
long benchmark(Func const& func) {
	auto t0 = std::chrono::high_resolution_clock::now();
	func();
	auto t1 = std::chrono::high_resolution_clock::now();
	auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
	return dt.count();
}

int main() {
	init();
	printf("Initial energy: %f\n", calc());  // Initial energy: -8.571527
	auto const dt = benchmark([&] {
		for (size_t i = 0; i < 100000; i++)
			step();
		});
	printf("Final energy: %f\n", calc());  // Final energy: -8.562095
	printf("Time elapsed: %ld ms\n", dt);  // 70ms
	//system("pause");
	return 0;
}
