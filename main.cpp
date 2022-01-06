#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <immintrin.h>
#include <utility>
#include <array>

const size_t STAR_NUM = 48;
constexpr size_t SIMD_WIDTH = 8;
constexpr size_t STAR_8_NUM = (STAR_NUM + 7) / 8;

const float G = 0.001;
const float eps = 0.001;
const float dt = 0.01;
const float G_dt = G * dt;

const __m256 eps2 = _mm256_set1_ps(eps * eps);
const __m256 half_G = _mm256_set1_ps(0.5 * G);
const __m256 vec_dt = _mm256_set1_ps(dt);
const __m256 vec_G_dt = _mm256_set1_ps(G_dt);

struct Star_8 {
	alignas(32) __m256 px;
	alignas(32) __m256 py;
	alignas(32) __m256 pz;
	alignas(32) __m256 mass;
	alignas(32) __m256 vx;
	alignas(32) __m256 vy;
	alignas(32) __m256 vz;
};

Star_8 stars[STAR_8_NUM];

static float frand() {
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
	for (int i = 0; i < STAR_8_NUM; i++) {
		for (int j = 0; j < 8; j++) {
			stars[i].px.m256_f32[j] = frand();
			stars[i].py.m256_f32[j] = frand();
			stars[i].pz.m256_f32[j] = frand();
			stars[i].vx.m256_f32[j] = frand();
			stars[i].vy.m256_f32[j] = frand();
			stars[i].vz.m256_f32[j] = frand();
			stars[i].mass.m256_f32[j] = frand() + 1;
		}
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
	std::array<__m256, STAR_8_NUM> d_vx{ 0.0f };
	std::array<__m256, STAR_8_NUM> d_vy{ 0.0f };
	std::array<__m256, STAR_8_NUM> d_vz{ 0.0f };
	for (size_t j = 0; j < STAR_8_NUM; j++) {
		auto& star_j = stars[j];
		for (size_t k = 0; k < 8; k++) {
			__m256 px_jk = _mm256_broadcast_ss(&star_j.px.m256_f32[k]);
			__m256 py_jk = _mm256_broadcast_ss(&star_j.py.m256_f32[k]);
			__m256 pz_jk = _mm256_broadcast_ss(&star_j.pz.m256_f32[k]);
			__m256 mass_jk = _mm256_broadcast_ss(&stars[j].mass.m256_f32[k]);
			/*__m256 d_vx = _mm256_setzero_ps();
			__m256 d_vy = _mm256_setzero_ps();
			__m256 d_vz = _mm256_setzero_ps();*/

			UNROLL<STAR_8_NUM>([&](size_t i) {  //unroll size: 6/3/2/1
				__m256 dx = _mm256_sub_ps(px_jk, stars[i].px);
				__m256 dy = _mm256_sub_ps(py_jk, stars[i].py);
				__m256 dz = _mm256_sub_ps(pz_jk, stars[i].pz);
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

				//__m256 mass_j = _mm256_load_ps(&mass[j]);
				__m256 factor = _mm256_mul_ps(inverse_sqrt, mass_jk);
				//float factor = mass[j] / sqrt(prod)^3;

				d_vx[i] = _mm256_fmadd_ps(dx, factor, d_vx[i]);
				d_vy[i] = _mm256_fmadd_ps(dy, factor, d_vy[i]);
				d_vz[i] = _mm256_fmadd_ps(dz, factor, d_vz[i]);
				//d_vx += dx * factor;
				//d_vy += dy * factor; 
				//d_vz += dz * factor; 
				});
		}
	}
	//for (size_t i = 0; i < STAR_NUM; i++) {
	//	px[i] += vx[i] * dt;
	//	py[i] += vy[i] * dt;
	//	pz[i] += vz[i] * dt;
	//}

	UNROLL<STAR_8_NUM>([&](size_t i) {   //unroll size: 6/3/2/1
		stars[i].vx = _mm256_fmadd_ps(d_vx[i], vec_G_dt, stars[i].vx);
		stars[i].vy = _mm256_fmadd_ps(d_vy[i], vec_G_dt, stars[i].vy);
		stars[i].vz = _mm256_fmadd_ps(d_vz[i], vec_G_dt, stars[i].vz);

		stars[i].px = _mm256_fmadd_ps(stars[i].vx, vec_dt, stars[i].px);
		stars[i].py = _mm256_fmadd_ps(stars[i].vy, vec_dt, stars[i].py);
		stars[i].pz = _mm256_fmadd_ps(stars[i].vz, vec_dt, stars[i].pz);
		//px[i] += vx[i] * dt;
		//py[i] += vy[i] * dt;
		//pz[i] += vz[i] * dt;
		});
}

float calc() {
	float energy = 0;
	for (size_t i = 0; i < STAR_8_NUM; i++) {
		auto const& star_i = stars[i];
		for (size_t k = 0; k < 8; k++) {
			__m256 px_ik = _mm256_broadcast_ss(&star_i.px.m256_f32[k]);
			__m256 py_ik = _mm256_broadcast_ss(&star_i.py.m256_f32[k]);
			__m256 pz_ik = _mm256_broadcast_ss(&star_i.pz.m256_f32[k]);
			float vx = stars[i].vx.m256_f32[k];
			float vy = stars[i].vy.m256_f32[k];
			float vz = stars[i].vz.m256_f32[k];
			float v2 = vx * vx + vy * vy + vz * vz;
			float mass_ik = stars[i].mass.m256_f32[k];
			__m256 mass_ik_256 = _mm256_broadcast_ss(&mass_ik);
			energy += mass_ik * v2 / 2;
			__m256 delta_e = _mm256_setzero_ps();
			for (size_t j = 0; j < STAR_8_NUM; j++) {
				__m256 dx = _mm256_sub_ps(stars[j].px, px_ik);

				//__m256 py_j = _mm256_load_ps(&py[j]);
				__m256 dy = _mm256_sub_ps(stars[j].py, py_ik);

				//__m256 pz_j = _mm256_load_ps(&pz[j]);
				__m256 dz = _mm256_sub_ps(stars[j].pz, pz_ik);
				//float dx = px[j] - px[i];
				//float dy = py[j] - py[i];
				//float dz = pz[j] - pz[i];
				__m256 prod = _mm256_mul_ps(dx, dx);
				prod = _mm256_fmadd_ps(dy, dy, prod);
				prod = _mm256_fmadd_ps(dz, dz, prod);
				prod = _mm256_add_ps(eps2, prod);
				prod = _mm256_rsqrt_ps(prod);
				prod = _mm256_mul_ps(mass_ik_256, prod);
				prod = _mm256_mul_ps(stars[j].mass, prod);
				//float d2 = dx * dx + dy * dy + dz * dz + eps * eps;
				delta_e = _mm256_add_ps(prod, delta_e); reduce256_add_ps(prod);
			}
			energy -= reduce256_add_ps(_mm256_mul_ps(half_G, delta_e));
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
