#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <immintrin.h>
#include <utility>
#include <array>

constexpr size_t STAR_NUM = 48;
constexpr size_t SIMD_WIDTH = 16;  //16 floats
constexpr size_t STAR_16_NUM = (STAR_NUM + SIMD_WIDTH - 1) / SIMD_WIDTH;

const float G = 0.001;
const float eps = 0.001;
const float dt = 0.01;
const float G_dt = G * dt;

const __m512 eps2 = _mm512_set1_ps(eps * eps);
const __m512 dt_vec = _mm512_set1_ps(dt);
const __m512 vec_G_dt = _mm512_set1_ps(G_dt);


struct Star_16 {
	alignas(64) __m512 px;
	alignas(64) __m512 py;
	alignas(64) __m512 pz;
	alignas(64) __m512 mass;
	alignas(64) __m512 vx;
	alignas(64) __m512 vy;
	alignas(64) __m512 vz;
};

std::array<Star_16, STAR_16_NUM> stars;

static float frand() {
	return (float)rand() / RAND_MAX * 2 - 1;
}

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

void init() {
	for (int i = 0; i < stars.size(); i++) {
		for (int j = 0; j < SIMD_WIDTH; j++) {
			stars[i].px.m512_f32[j] = frand();
			stars[i].py.m512_f32[j] = frand();
			stars[i].pz.m512_f32[j] = frand();
			stars[i].vx.m512_f32[j] = frand();
			stars[i].vy.m512_f32[j] = frand();
			stars[i].vz.m512_f32[j] = frand();
			stars[i].mass.m512_f32[j] = frand() + 1;
		}
	}
}

template<class Fn, size_t... M>
static void unroll_impl(Fn fn, size_t L, std::integer_sequence<size_t, M...> iter) {
	constexpr auto S = sizeof...(M);
	if (L == 1) {
		((fn(M)), ...);
	}
	else {
		for (size_t i = 0; i < L; i++) {
			((fn(M + i * S)), ...);
		}
	}
}

template<size_t N, size_t S = N, class Fn>   // N: total iterations, S = iterations per loop. S==N implies unrolling completely
constexpr static void UNROLL(Fn fn) {
	static_assert(N % S == 0);
	unroll_impl(fn, N / S, std::make_index_sequence<S>());
}

void step() {
	
	UNROLL<stars.size(), 1>([&](size_t j) {
		auto& star_j = stars[j];
		UNROLL<SIMD_WIDTH, 1>([&](size_t k) {		
			__m512 px_jk = _mm512_set1_ps(star_j.px.m512_f32[k]);  
			__m512 py_jk = _mm512_set1_ps(star_j.py.m512_f32[k]);
			__m512 pz_jk = _mm512_set1_ps(star_j.pz.m512_f32[k]);
			__m512 mass_jk = _mm512_set1_ps(star_j.mass.m512_f32[k]);

			//__m512 d_vx = _mm512_setzero_ps();  //d_vx = 0
			//__m512 d_vy = _mm512_setzero_ps();	//d_vy = 0
			//__m512 d_vz = _mm512_setzero_ps();	//d_vz = 0

			UNROLL<stars.size()>([&](size_t i) {  //unrolling size: 3/1
				__m512 dx = _mm512_sub_ps(px_jk, stars[i].px);
				__m512 dy = _mm512_sub_ps(py_jk, stars[i].py);
				__m512 dz = _mm512_sub_ps(pz_jk, stars[i].pz);
				//float dx = px[j] - px[i];
				//float dy = py[j] - py[i];
				//float dz = pz[j] - pz[i];

				__m512 prod = _mm512_mul_ps(dx, dx);
				prod = _mm512_fmadd_ps(dy, dy, prod);
				prod = _mm512_fmadd_ps(dz, dz, prod);
				prod = _mm512_add_ps(eps2, prod);
				//float prod = dx * dx + dy * dy + dz * dz + eps * eps;

				__m512 inverse_sqrt = _mm512_rsqrt14_ps(prod);  // an approximate version of _mm512_invsqrt_ps 
				__m512 inverse_sqrt2 = _mm512_mul_ps(inverse_sqrt, inverse_sqrt);
				__m512 inverse_sqrt3 = _mm512_mul_ps(inverse_sqrt2, inverse_sqrt);
				//float inverse_sqrt = 1 / sqrt(prod)^3;

				__m512 factor = _mm512_mul_ps(inverse_sqrt3, mass_jk);
				factor = _mm512_mul_ps(vec_G_dt, factor);
				//float factor = mass[j] / sqrt(prod)^3;

				/*d_vx = _mm512_fmadd_ps(dx, factor, d_vx);
				d_vy = _mm512_fmadd_ps(dy, factor, d_vy);
				d_vz = _mm512_fmadd_ps(dz, factor, d_vz);*/

				stars[i].vx = _mm512_fmadd_ps(dx, factor, stars[i].vx);
				stars[i].vy = _mm512_fmadd_ps(dy, factor, stars[i].vy);
				stars[i].vz = _mm512_fmadd_ps(dz, factor, stars[i].vz);

		/*		_mm512_store_ps(&vx[i], new_vx_i);
				_mm512_store_ps(&vy[i], new_vy_i);
				_mm512_store_ps(&vz[i], new_vz_i);*/
				//d_vx += dx * factor;
				//d_vy += dy * factor; 
				//d_vz += dz * factor; 
				});

			//star_i.vx.m512_f32[k] += reduce512_add_ps(d_vx) * G_dt;
			//star_i.vy.m512_f32[k] += reduce512_add_ps(d_vy) * G_dt;
			//star_i.vz.m512_f32[k] += reduce512_add_ps(d_vz) * G_dt;
			//star_i.vx.m512_f32[k] += _mm512_reduce_add_ps(d_vx) * G_dt;  //alternatives
			//star_i.vy.m512_f32[k] += _mm512_reduce_add_ps(d_vy) * G_dt;
			//star_i.vz.m512_f32[k] += _mm512_reduce_add_ps(d_vz) * G_dt;
			});
		});

	UNROLL<stars.size()>([&](size_t i) {   //unrolling size: 3/1
		stars[i].px = _mm512_fmadd_ps(stars[i].vx, dt_vec, stars[i].px);
		stars[i].py = _mm512_fmadd_ps(stars[i].vy, dt_vec, stars[i].py);
		stars[i].pz = _mm512_fmadd_ps(stars[i].vz, dt_vec, stars[i].pz);
		//px[i] += vx[i] * dt;
		//py[i] += vy[i] * dt;
		//pz[i] += vz[i] * dt;
		});
}

float calc() {
	float energy = 0.f;
	const float half_G = 0.5 * G;
	for (size_t i = 0; i < stars.size(); i++) {
		auto const& star_i = stars[i];
		for (size_t k = 0; k < SIMD_WIDTH; k++) {

			const float vx = star_i.vx.m512_f32[k];
			const float vy = star_i.vy.m512_f32[k];
			const float vz = star_i.vz.m512_f32[k];
			const float v2 = vx * vx + vy * vy + vz * vz;
			float mass_ik = star_i.mass.m512_f32[k];
			energy += mass_ik * v2 / 2;
			float delta_e = 0.f;

			for (size_t j = 0; j < stars.size(); j++) {
				auto const& star_j = stars[j];
				for (size_t m = 0; m < SIMD_WIDTH; m++) {

					if (i != j && k != m) {
						const float dx = star_j.px.m512_f32[m];
						const float dy = star_j.py.m512_f32[m];
						const float dz = star_j.pz.m512_f32[m];
						const float d2 = dx * dx + dy * dy + dz * dz + eps * eps;
						delta_e += star_j.mass.m512_f32[m] / sqrt(d2);
					}
				}
			}
			energy -= delta_e * mass_ik * half_G;
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
		for (size_t i = 0; i < 1000000; i++)
			step();
		});
	printf("Final energy: %f\n", calc());  // Final energy: -8.562095
	printf("Time elapsed: %lld ms\n", dt);
	//system("pause");
	return 0;
}
