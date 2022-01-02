#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <cmath>
#include <immintrin.h>

const size_t STAR_NUM = 48;

float frand() {
	return (float)rand() / RAND_MAX * 2 - 1;
}

// x = ( x7, x6, x5, x4, x3, x2, x1, x0 )
float sum8(__m256 x) {
	// hiQuad = ( x7, x6, x5, x4 )
	const __m128 hiQuad = _mm256_extractf128_ps(x, 1);
	// loQuad = ( x3, x2, x1, x0 )
	const __m128 loQuad = _mm256_castps256_ps128(x);
	// sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
	const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
	// loDual = ( -, -, x1 + x5, x0 + x4 )
	const __m128 loDual = sumQuad;
	// hiDual = ( -, -, x3 + x7, x2 + x6 )
	const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
	// sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
	const __m128 sumDual = _mm_add_ps(loDual, hiDual);
	// lo = ( -, -, -, x0 + x2 + x4 + x6 )
	const __m128 lo = sumDual;
	// hi = ( -, -, -, x1 + x3 + x5 + x7 )
	const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
	// sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
	const __m128 sum = _mm_add_ss(lo, hi);
	return _mm_cvtss_f32(sum);
}

alignas(4) float px[STAR_NUM];
alignas(4) float py[STAR_NUM];
alignas(4) float pz[STAR_NUM];
alignas(4) float vx[STAR_NUM];
alignas(4) float vy[STAR_NUM];
alignas(4) float vz[STAR_NUM];
alignas(4) float mass[STAR_NUM];




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


const float G_dt = G * dt;

void step() {
	const __m256 eps2 = _mm256_set1_ps(eps * eps);
	const __m256 dt_vec = _mm256_set1_ps(dt);

	__m256 dvx, dvy, dvz;

	for (size_t i = 0; i < STAR_NUM; i++) {
		__m256 px_i = _mm256_broadcast_ss(&px[i]);
		__m256 py_i = _mm256_broadcast_ss(&py[i]);
		__m256 pz_i = _mm256_broadcast_ss(&pz[i]);


		dvx = _mm256_xor_ps(dvx, dvx);//ax=(0,0,0,0,0,0,0,0)
		dvy = _mm256_xor_ps(dvy, dvy);
		dvz = _mm256_xor_ps(dvz, dvz);

		for (size_t j = 0; j < STAR_NUM; j += 8) {
			__m256 px_j = _mm256_load_ps(&px[j]);
			__m256 py_j = _mm256_load_ps(&py[j]);
			__m256 pz_j = _mm256_load_ps(&pz[j]);

			__m256 dx = _mm256_sub_ps(px_j, px_i);
			__m256 dy = _mm256_sub_ps(py_j, py_i);
			__m256 dz = _mm256_sub_ps(pz_j, pz_i);
			//float dx = px[j] - px[i];
			//float dy = py[j] - py[i];
			//float dz = pz[j] - pz[i];
			__m256 prod = _mm256_mul_ps(dx, dx);
			prod = _mm256_fmadd_ps(dy, dy, prod);
			prod = _mm256_fmadd_ps(dz, dz, prod);
			prod = _mm256_add_ps(eps2, prod);
			//float d2 = dx * dx + dy * dy + dz * dz + eps * eps;
			__m256 inverse_sqrt = _mm256_rsqrt_ps(prod);
			__m256 inverse_sqrt2 = _mm256_mul_ps(inverse_sqrt, inverse_sqrt);
			__m256 inverse_sqrt3 = _mm256_mul_ps(inverse_sqrt2, inverse_sqrt);//r2*r2*r2
			//d2 *= sqrt(d2);

			__m256 mass_j = _mm256_broadcast_ss(&mass[j]);
			__m256 factor = _mm256_mul_ps(inverse_sqrt3, mass_j);

			dvx = _mm256_fmadd_ps(dx, factor, dvx);
			dvy = _mm256_fmadd_ps(dy, factor, dvy);
			dvz = _mm256_fmadd_ps(dz, factor, dvz);
		}
		vx[i] += sum8(dvx) * G_dt;
		vy[i] += sum8(dvy) * G_dt;
		vz[i] += sum8(dvz) * G_dt;
	}

	//for (size_t i = 0; i < STAR_NUM; i++) {
	//	px[i] += vx[i] * dt;
	//	py[i] += vy[i] * dt;
	//	pz[i] += vz[i] * dt;
	//}

	for (size_t i = 0; i < STAR_NUM; i += 8) {
		__m256 px_j = _mm256_load_ps(&px[i]);
		__m256 py_j = _mm256_load_ps(&py[i]);
		__m256 pz_j = _mm256_load_ps(&pz[i]);

		__m256 vx_j = _mm256_load_ps(&vx[i]);
		__m256 vy_j = _mm256_load_ps(&vy[i]);
		__m256 vz_j = _mm256_load_ps(&vz[i]);

		_mm256_store_ps(&px[i], _mm256_fmadd_ps(vx_j, dt_vec, px_j));
		_mm256_store_ps(&py[i], _mm256_fmadd_ps(vy_j, dt_vec, py_j));
		_mm256_store_ps(&pz[i], _mm256_fmadd_ps(vz_j, dt_vec, pz_j));
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
	auto dt = benchmark([&] {
		for (size_t i = 0; i < 1000000; i++)
			step();
		});
	printf("Final energy: %f\n", calc());  // Final energy: -8.511734
	printf("Time elapsed: %ld ms\n", dt);
	system("pause");
	return 0;
}
