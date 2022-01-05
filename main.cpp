#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <immintrin.h>
#include <utility>

const size_t STAR_NUM = 48;

const float G = 0.001;
const float eps = 0.001;
const float dt = 0.01;
const float G_dt = G * dt;

const __m256 vec_eps2 = _mm256_set1_ps(eps * eps);
const __m256 vec_dt = _mm256_set1_ps(dt);
const __m256 vec_G_dt = _mm256_set1_ps(G_dt);

alignas(32) float px[STAR_NUM];
alignas(32) float py[STAR_NUM];
alignas(32) float pz[STAR_NUM];
alignas(32) float vx[STAR_NUM];
alignas(32) float vy[STAR_NUM];
alignas(32) float vz[STAR_NUM];
alignas(32) float mass[STAR_NUM];

static float frand() {
	return (float)rand() / RAND_MAX * 2 - 1;
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

template<class Fn, size_t... M>
static void unroll_impl(Fn fn, size_t L, std::integer_sequence<size_t, M...> iter) {
	constexpr auto S = sizeof...(M);
	for (size_t i = 0; i < L; i++) {
		((fn(M + i * S)), ...);
	}
}

template<size_t N, size_t S = N, class Fn>   // N: total iterations, S = iterations per loop. S==N implies unrolling completely
constexpr static void UNROLL(Fn fn) {
	static_assert(N % S == 0);
	unroll_impl(fn, N / S, std::make_index_sequence<S>());
}

void step() {
	

	for (size_t j = 0; j < STAR_NUM; j++) {
		__m256 px_j = _mm256_broadcast_ss(&px[j]);
		__m256 py_j = _mm256_broadcast_ss(&py[j]);
		__m256 pz_j = _mm256_broadcast_ss(&pz[j]);
		__m256 mass_j = _mm256_load_ps(&mass[j]);

	/*	__m256 d_vx = _mm256_setzero_ps();
		__m256 d_vy = _mm256_setzero_ps();
		__m256 d_vz = _mm256_setzero_ps();*/

		UNROLL<6, 6>([&](size_t iter) {  //unroll size: 6/3/2/1
			const size_t i = iter * 8;
			__m256 px_i = _mm256_load_ps(&px[i]);
			__m256 dx = _mm256_sub_ps(px_j, px_i);

			__m256 py_i = _mm256_load_ps(&py[i]);
			__m256 dy = _mm256_sub_ps(py_j, py_i);

			__m256 pz_i = _mm256_load_ps(&pz[i]);
			__m256 dz = _mm256_sub_ps(pz_j, pz_i);
			//float dx = px[j] - px[i];
			//float dy = py[j] - py[i];
			//float dz = pz[j] - pz[i];

			__m256 prod = _mm256_mul_ps(dx, dx);
			prod = _mm256_fmadd_ps(dy, dy, prod);
			prod = _mm256_fmadd_ps(dz, dz, prod);
			prod = _mm256_add_ps(vec_eps2, prod);
			//float prod = dx * dx + dy * dy + dz * dz + eps * eps;

			__m256 inverse_sqrt = _mm256_rsqrt_ps(prod);
			__m256 inverse_sqrt2 = _mm256_mul_ps(inverse_sqrt, inverse_sqrt);
			inverse_sqrt = _mm256_mul_ps(inverse_sqrt2, inverse_sqrt);
			//float inverse_sqrt = 1 / sqrt(prod)^3;

			
			__m256 factor = _mm256_mul_ps(inverse_sqrt, mass_j);
			factor = _mm256_mul_ps(vec_G_dt, factor);
			//float factor = mass[j] / sqrt(prod)^3;

			__m256 vx_i = _mm256_load_ps(&vx[i]);
			__m256 vy_i = _mm256_load_ps(&vy[i]);
			__m256 vz_i = _mm256_load_ps(&vz[i]);

			__m256 new_vx_i = _mm256_fmadd_ps(dx, factor, vx_i);
			__m256 new_vy_i = _mm256_fmadd_ps(dy, factor, vy_i);
			__m256 new_vz_i = _mm256_fmadd_ps(dz, factor, vz_i);

		/*	d_vx = _mm256_fmadd_ps(dy, factor, d_vy);
			d_vy = _mm256_fmadd_ps(dy, factor, d_vy);
			d_vz = _mm256_fmadd_ps(dz, factor, d_vz);*/
			//d_vx += dx * factor;
			//d_vy += dy * factor; 
			//d_vz += dz * factor; 
			
			_mm256_store_ps(&vx[i], new_vx_i);
			_mm256_store_ps(&vy[i], new_vy_i);
			_mm256_store_ps(&vz[i], new_vz_i);
			});

		//vx[i] += reduce256_add_ps(d_vx) * G_dt;
		//vy[i] += reduce256_add_ps(d_vy) * G_dt;
		//vz[i] += reduce256_add_ps(d_vz) * G_dt;
	}
	//for (size_t i = 0; i < STAR_NUM; i++) {
	//	px[i] += vx[i] * dt;
	//	py[i] += vy[i] * dt;
	//	pz[i] += vz[i] * dt;
	//}

	UNROLL<6, 6>([&](size_t iter) {   //unroll size: 6/3/2/1
		const size_t i = iter * 8;
		__m256 px_i = _mm256_load_ps(&px[i]);
		__m256 vx_i = _mm256_load_ps(&vx[i]);
		_mm256_store_ps(&px[i], _mm256_fmadd_ps(vx_i, vec_dt, px_i));

		__m256 py_i = _mm256_load_ps(&py[i]);
		__m256 vy_i = _mm256_load_ps(&vy[i]);
		_mm256_store_ps(&py[i], _mm256_fmadd_ps(vy_i, vec_dt, py_i));

		__m256 pz_i = _mm256_load_ps(&pz[i]);
		__m256 vz_i = _mm256_load_ps(&vz[i]);
		_mm256_store_ps(&pz[i], _mm256_fmadd_ps(vz_i, vec_dt, pz_i));
		//px[i] += vx[i] * dt;
		//py[i] += vy[i] * dt;
		//pz[i] += vz[i] * dt;
		});
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
long long benchmark(Func const& func) {
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
	printf("Time elapsed: %lld ms\n", dt);
	//system("pause");
	return 0;
}
