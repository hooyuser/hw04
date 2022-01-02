#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <cmath>
#include <immintrin.h>

const size_t STAR_NUM = 48 * 4;

float frand() {
	return (float)rand() / RAND_MAX * 2 - 1;
}
/*
//x = ( x7, x6, x5, x4, x3, x2, x1, x0 )
float sum16(__m512 zmm) {
	__m256 high = _mm512_extractf32x8_ps(zmm, 1);
	__m256 low = _mm512_castps512_ps256(zmm);
	__m256 sumOct = _mm256_add_ps(low, high);
	// hiQuad = ( x7, x6, x5, x4 )
	const __m128 hiQuad = _mm256_extractf128_ps(sumOct, 1);
	// loQuad = ( x3, x2, x1, x0 )
	const __m128 loQuad = _mm256_castps256_ps128(sumOct);
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
}*/

static float reduce512_add_ps(__m512 zmm) {  
	__m256 high = _mm512_extractf32x8_ps(zmm, 1);
	__m256 low = _mm512_castps512_ps256(zmm);
	__m256 sumOct = _mm256_add_ps(low, high);

	__m128 vlow = _mm256_castps256_ps128(sumOct);
	// vlow = ( x3, x2, x1, x0 )

	__m128 vhigh = _mm256_extractf128_ps(sumOct, 1);
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

alignas(64) float px[STAR_NUM];
alignas(64) float py[STAR_NUM];
alignas(64) float pz[STAR_NUM];
alignas(64) float vx[STAR_NUM];
alignas(64) float vy[STAR_NUM];
alignas(64) float vz[STAR_NUM];
alignas(64) float mass[STAR_NUM];




//Star<STAR_NUM> stars;

void init() {
	for (size_t i = 0; i < STAR_NUM; i++) {
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

const __m512 eps2 = _mm512_set1_ps(eps * eps);
const __m512 dt_vec = _mm512_set1_ps(dt);
const float G_dt = G * dt;

void step() {
	

	for (size_t i = 0; i < STAR_NUM; i++) {
		alignas(64) __m512 px_i = _mm512_set1_ps(px[i]);
		alignas(64) __m512 py_i = _mm512_set1_ps(py[i]);
		alignas(64) __m512 pz_i = _mm512_set1_ps(pz[i]);

		//alignas(64) __m512 dvx, dvy, dvz;
		__m512 dvx = _mm512_setzero_ps();//ax=(0,0,0,0,0,0,0,0)
		__m512 dvy = _mm512_setzero_ps();
		__m512 dvz = _mm512_setzero_ps();


#pragma unroll 6
		for (size_t j = 0; j < STAR_NUM; j += 16) {
			__m512 px_j = _mm512_load_ps(&px[j]);
			__m512 dx = _mm512_sub_ps(px_j, px_i);

			__m512 py_j = _mm512_load_ps(&py[j]);
			__m512 dy = _mm512_sub_ps(py_j, py_i);

			__m512 pz_j = _mm512_load_ps(&pz[j]);
			__m512 dz = _mm512_sub_ps(pz_j, pz_i);
			//float dx = px[j] - px[i];
			//float dy = py[j] - py[i];
			//float dz = pz[j] - pz[i];
			__m512 prod = _mm512_mul_ps(dx, dx);
			prod = _mm512_fmadd_ps(dy, dy, prod);
			prod = _mm512_fmadd_ps(dz, dz, prod);
			prod = _mm512_add_ps(eps2, prod);
			//float prod = dx * dx + dy * dy + dz * dz + eps * eps;
			__m512 inverse_sqrt = _mm512_invsqrt_ps(prod);
			__m512 inverse_sqrt2 = _mm512_mul_ps(inverse_sqrt, inverse_sqrt);
			__m512 inverse_sqrt3 = _mm512_mul_ps(inverse_sqrt2, inverse_sqrt);//r2*r2*r2
			//d2 *= sqrt(d2);

			__m512 mass_j = _mm512_load_ps(&mass[j]);
			__m512 factor = _mm512_mul_ps(inverse_sqrt3, mass_j);

			dvx = _mm512_fmadd_ps(dx, factor, dvx);
			dvy = _mm512_fmadd_ps(dy, factor, dvy);
			dvz = _mm512_fmadd_ps(dz, factor, dvz);
		}
		vx[i] += reduce512_add_ps(dvx) * G_dt;
		vy[i] += reduce512_add_ps(dvy) * G_dt;
		vz[i] += reduce512_add_ps(dvz) * G_dt;
		//vx[i] += _mm512_reduce_add_ps(dvx) * G_dt;
		//vy[i] += _mm512_reduce_add_ps(dvy) * G_dt;
		//vz[i] += _mm512_reduce_add_ps(dvz) * G_dt;
	}

	for (size_t i = 0; i < STAR_NUM; i += 16) {
		__m512 px_j = _mm512_load_ps(&px[i]);
		__m512 py_j = _mm512_load_ps(&py[i]);
		__m512 pz_j = _mm512_load_ps(&pz[i]);

		__m512 vx_j = _mm512_load_ps(&vx[i]);
		__m512 vy_j = _mm512_load_ps(&vy[i]);
		__m512 vz_j = _mm512_load_ps(&vz[i]);

		_mm512_store_ps(&px[i], _mm512_fmadd_ps(vx_j, dt_vec, px_j));
		_mm512_store_ps(&py[i], _mm512_fmadd_ps(vy_j, dt_vec, py_j));
		_mm512_store_ps(&pz[i], _mm512_fmadd_ps(vz_j, dt_vec, pz_j));
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
		for (size_t i = 0; i < 100000; i++)
			step();
		});
	printf("Final energy: %f\n", calc());  // Final energy: -8.511734
	printf("Time elapsed: %ld ms\n", dt);
	system("pause");
	return 0;
}
