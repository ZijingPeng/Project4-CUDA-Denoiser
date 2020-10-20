#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1
#define STREAM_COMPACTION 1
#define SORT_BY_MATERIAL 0
#define CACHE_ENABLE 1
#define PROFILE_ENABLE 0
#define DEPTH_OF_FIELD_ENABLE 0
#define ANTIALIASING 0
#define MOTION_BLUR_ENABLE 0
#define AMBIENT_LIGHT_ENABLE 0
#define SHOW_GBUFFER 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line) {
#if ERRORCHECK
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess == err) {
		return;
	}

	fprintf(stderr, "CUDA error");
	if (file) {
		fprintf(stderr, " (%s:%d)", file, line);
	}
	fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
	getchar();
#  endif
	exit(EXIT_FAILURE);
#endif
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
	int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
	return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
	int iter, glm::vec3* image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		glm::vec3 pix = image[index];

		glm::ivec3 color;
		color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
		color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
		color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

		// Each thread writes one pixel location in the texture (textel)
		pbo[index].w = 0;
		pbo[index].x = color.x;
		pbo[index].y = color.y;
		pbo[index].z = color.z;
	}
}

__global__ void gbufferToPBO(uchar4* pbo, glm::ivec2 resolution, GBufferPixel* gBuffer) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
#if SHOW_GBUFFER == 0
		glm::vec3 normal = glm::normalize(gBuffer[index].normal) * 255.0f;

		pbo[index].w = 0;
		pbo[index].x = abs(normal.x);
		pbo[index].y = abs(normal.y);
		pbo[index].z = abs(normal.z);
#elif SHOW_GBUFFER == 1
		float scaler = 25.0;
		glm::vec3 pos = gBuffer[index].pos * scaler;
		pos = glm::abs(pos);
		pos = glm::clamp(pos, 0.0f, 255.0f);

		pbo[index].w = 0;
		pbo[index].x = pos.x;
		pbo[index].y = pos.y;
		pbo[index].z = pos.z;
#endif
	}
}
__global__ void initDenoisedImage(glm::vec3* image, glm::vec3* denoisedImage, glm::ivec2 resolution, int iter)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		glm::vec3 pix = image[index];
		pix.x = glm::clamp(pix.x / iter, 0.f, 1.f);
		pix.y = glm::clamp(pix.y / iter, 0.f, 1.f);
		pix.z = glm::clamp(pix.z / iter, 0.f, 1.f);
		denoisedImage[index] = pix;
	}
}

static Scene* hst_scene = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static ShadeableIntersection* dev_intersections_cache = NULL;
static Triangle* dev_triangles = NULL;
static GBufferPixel* dev_gBuffer = NULL;
static glm::vec3* dev_denoised_image = NULL;
static glm::vec3* dev_denoised_image2 = NULL;
static float* dev_kernel = NULL;


static float host_kernel[] =
{
	0.003765,	0.015019,	0.023792,	0.015019,	0.003765,
	0.015019,	0.059912,	0.094907,	0.059912,	0.015019,
	0.023792,	0.094907,	0.150342,	0.094907,	0.023792,
	0.015019,	0.059912,	0.094907,	0.059912,	0.015019,
	0.003765,	0.015019,	0.023792,	0.015019,	0.003765
};

// variables for profiling
cudaEvent_t start, stop;
float totalTime = 0.0;
bool countStart = true;

void pathtraceInit(Scene* scene) {
	hst_scene = scene;
	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Triangle));
	cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	cudaMalloc(&dev_gBuffer, pixelcount * sizeof(GBufferPixel));

	cudaMalloc(&dev_denoised_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_denoised_image, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_denoised_image2, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_denoised_image2, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_kernel, 10 * 10 * sizeof(float));
	cudaMemcpy(dev_kernel, host_kernel, 25 * sizeof(float), cudaMemcpyHostToDevice);

#if CACHE_ENABLE
	cudaMalloc(&dev_intersections_cache, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections_cache, 0, pixelcount * sizeof(ShadeableIntersection));
#endif // CACHE_ENABLE

	// TODO: initialize any extra device memeory you need

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	checkCUDAError("pathtraceInit");
}

void pathtraceFree() {

	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths);
	cudaFree(dev_geoms);
	cudaFree(dev_triangles);
	cudaFree(dev_materials);
	cudaFree(dev_intersections);
	cudaFree(dev_gBuffer);
	cudaFree(dev_denoised_image);
	cudaFree(dev_denoised_image2);
	cudaFree(dev_kernel);
#if CACHE_ENABLE
	cudaFree(dev_intersections_cache);
#endif // CACHE_ENABLE
	// TODO: clean up any extra device memory you created

	checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment& segment = pathSegments[index];

		segment.ray.origin = cam.position;

		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

		// antialiasing
#if ANTIALIASING
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
		thrust::uniform_real_distribution<float> u01(-0.5, 0.5);
		// add a small offset
		x += u01(rng);
		y += u01(rng);
#endif
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
			);

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;

#if DEPTH_OF_FIELD_ENABLE || MOTION_BLUR_ENABLE
		//thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
#endif

#if DEPTH_OF_FIELD_ENABLE
		// depth of field
		float lensRadius = 0.05f;
		float focalDistance = 12.0f;

		float p0 = u01(rng);
		float p1 = u01(rng);
		// sample a point from lens
		segment.ray.origin = cam.position + p0 * lensRadius * cam.up + p1 * lensRadius * cam.right;
		float asp = focalDistance / glm::length(cam.view);
		segment.ray.direction = cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f);
		glm::vec3 target = cam.position + segment.ray.direction * asp;
		segment.ray.direction = glm::normalize(target - segment.ray.origin);
#endif

#if MOTION_BLUR_ENABLE
		thrust::uniform_real_distribution<float> u02(0.0f, 1.0f);
		segment.ray.time = u02(rng);
#endif
	}
}

__global__ void computeIntersections(
	int depth
	, int num_paths
	, PathSegment* pathSegments
	, Geom* geoms
	, int geoms_size
	, ShadeableIntersection* intersections
	, Triangle* triangles
	)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;

		// naive parse through global geoms

		for (int i = 0; i < geoms_size; i++)
		{
			Geom& geom = geoms[i];

			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == MESH)
			{
				t = meshIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside, triangles);
			}
			// TODO: add more intersection tests here... triangle? metaball? CSG?

			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_geom_index = i;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
			}
		}

		if (hit_geom_index == -1)
		{
			intersections[path_index].t = -1.0f;
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
			intersections[path_index].point = intersect_point;
		}
	}
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeFakeMaterial(
	int iter
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, Material* materials
	)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		ShadeableIntersection intersection = shadeableIntersections[idx];
		if (intersection.t > 0.0f) { // if the intersection exists...
		  // Set up the RNG
		  // LOOK: this is how you use thrust's RNG! Please look at
		  // makeSeededRandomEngine as well.
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
			thrust::uniform_real_distribution<float> u01(0, 1);

			Material material = materials[intersection.materialId];
			glm::vec3 materialColor = material.color;

			// If the material indicates that the object was a light, "light" the ray
			if (material.emittance > 0.0f) {
				pathSegments[idx].color *= (materialColor * material.emittance);
			}
			// Otherwise, do some pseudo-lighting computation. This is actually more
			// like what you would expect from shading in a rasterizer like OpenGL.
			// TODO: replace this! you should be able to start with basically a one-liner
			else {
				//float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
				//pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
				//pathSegments[idx].color *= u01(rng); // apply some noise because why not

				scatterRay(pathSegments[idx], intersection.point, intersection.surfaceNormal, material, rng);
			}
			// If there was no intersection, color the ray black.
			// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
			// used for opacity, in which case they can indicate "no opacity".
			// This can be useful for post-processing and image compositing.
		}
		else {
			pathSegments[idx].color = glm::vec3(0.0f);
		}
	}
}

__global__ void shadeBSDFMaterial(
	int iter
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, Material* materials
	)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		ShadeableIntersection intersection = shadeableIntersections[idx];
		if (pathSegments[idx].remainingBounces <= 0) {
			return;
		}
		if (intersection.t > 0.0f) { // if the intersection exists...

			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegments[idx].remainingBounces);

			Material material = materials[intersection.materialId];
			glm::vec3 materialColor = material.color;

			// If the material indicates that the object was a light, "light" the ray
			if (material.emittance > 0.0f) {
				if (pathSegments[idx].remainingBounces >= 0) {
					pathSegments[idx].color *= (materialColor * material.emittance);
					pathSegments[idx].remainingBounces = -1;
				}
			}
			else {
				scatterRay(pathSegments[idx], intersection.point, intersection.surfaceNormal, material, rng);
				pathSegments[idx].remainingBounces--;
				// if the last bounce is not the light source, it should not be shaded
				if (pathSegments[idx].remainingBounces <= 0) {
#if AMBIENT_LIGHT_ENABLE
					float t = 0.5 * (glm::normalize(pathSegments[idx].ray.direction).y + 1.0);
					pathSegments[idx].color *= (1.0f - t) * glm::vec3(1.0, 1.0, 1.0) + t * glm::vec3(0.5, 0.7, 1.0);
#else
					pathSegments[idx].color = glm::vec3(0.0f);
#endif // AMBIENT_LIGHT_ENABLE	
				}
			}
		}
		else {
#if AMBIENT_LIGHT_ENABLE
			float t = 0.5 * (glm::normalize(pathSegments[idx].ray.direction).y + 1.0);
			pathSegments[idx].color *= (1.0f - t) * glm::vec3(1.0, 1.0, 1.0) + t * glm::vec3(0.5, 0.7, 1.0);
#else
			pathSegments[idx].color = glm::vec3(0.0f);
#endif // AMBIENT_LIGHT_ENABLE
			pathSegments[idx].remainingBounces = 0;
		}
	}
}

__global__ void generateGBuffer(
	int num_paths,
	ShadeableIntersection* shadeableIntersections,
	PathSegment* pathSegments,
	GBufferPixel* gBuffer) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		gBuffer[idx].normal = shadeableIntersections[idx].surfaceNormal;
		gBuffer[idx].pos = shadeableIntersections[idx].point;
	}

}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] += iterationPath.color;
	}
}


struct is_zero
{
	__host__ __device__
		bool operator()(const PathSegment p)
	{
		return p.remainingBounces > 0;
	}
};



struct material_cmp {
	__host__ __device__ bool operator()(const ShadeableIntersection& s1, const ShadeableIntersection& s2) {
		return s1.materialId < s2.materialId;
	}
};

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(int frame, int iter) {
	const int traceDepth = hst_scene->state.traceDepth;
	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
	(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 128;

	///////////////////////////////////////////////////////////////////////////

	// Recap:
	// * Initialize array of path rays (using rays that come out of the camera)
	//   * You can pass the Camera object to that kernel.
	//   * Each path ray must carry at minimum a (ray, color) pair,
	//   * where color starts as the multiplicative identity, white = (1, 1, 1).
	//   * This has already been done for you.
	// * For each depth:
	//   * Compute an intersection in the scene for each path ray.
	//     A very naive version of this has been implemented for you, but feel
	//     free to add more primitives and/or a better algorithm.
	//     Currently, intersection distance is recorded as a parametric distance,
	//     t, or a "distance along the ray." t = -1.0 indicates no intersection.
	//     * Color is attenuated (multiplied) by reflections off of any object
	//   * TODO: Stream compact away all of the terminated paths.
	//     You may use either your implementation or `thrust::remove_if` or its
	//     cousins.
	//     * Note that you can't really use a 2D kernel launch any more - switch
	//       to 1D.
	//   * TODO: Shade the rays that intersected something or didn't bottom out.
	//     That is, color the ray by performing a color computation according
	//     to the shader, then generate a new ray to continue the ray path.
	//     We recommend just updating the ray's PathSegment in place.
	//     Note that this step may come before or after stream compaction,
	//     since some shaders you write may also cause a path to terminate.
	// * Finally, add this iteration's results to the image. This has been done
	//   for you.

	// perform one iteration of path tracing

	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);

	checkCUDAError("generate camera ray");

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;
	int remaining_paths = num_paths;

#if PROFILE_ENABLE
	cudaEventRecord(start);
#endif // PROFILE_ENABLE

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	 // Empty gbuffer
	cudaMemset(dev_gBuffer, 0, pixelcount * sizeof(GBufferPixel));

	bool iterationComplete = false;
	while (!iterationComplete) {


		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

		dim3 numblocksPathSegmentTracing = (remaining_paths + blockSize1d - 1) / blockSize1d;
#if CACHE_ENABLE && !ANTIALIASING
		if (depth <= 0) {
			if (iter > 1) {
				// tracing
				cudaMemcpy(dev_intersections, dev_intersections_cache, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
			}
			else {
				computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
					depth
					, remaining_paths
					, dev_paths
					, dev_geoms
					, hst_scene->geoms.size()
					, dev_intersections
					, dev_triangles
					);
				checkCUDAError("trace one bounce");
				cudaMemcpy(dev_intersections_cache, dev_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
			}
		}
		else {
			// tracing
			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth
				, remaining_paths
				, dev_paths
				, dev_geoms
				, hst_scene->geoms.size()
				, dev_intersections
				, dev_triangles
				);
			checkCUDAError("trace one bounce");
		}
#else
		// tracing
		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth
			, remaining_paths
			, dev_paths
			, dev_geoms
			, hst_scene->geoms.size()
			, dev_intersections
			, dev_triangles
			);
		checkCUDAError("trace one bounce");
#endif // CACHE_ENABLE
		cudaDeviceSynchronize();

		if (depth == 0) {
			generateGBuffer << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, dev_intersections, dev_paths, dev_gBuffer);
		}
		depth++;

#if SORT_BY_MATERIAL
		// sort by material id
		thrust::device_ptr<ShadeableIntersection> pIntersection = thrust::device_pointer_cast<ShadeableIntersection>(dev_intersections);
		thrust::device_ptr<PathSegment> pPathSegment = thrust::device_pointer_cast<PathSegment>(dev_paths);
		thrust::sort_by_key(pIntersection, pIntersection + remaining_paths, pPathSegment, material_cmp());
#endif // SORT_BY_MATERIAL

		// TODO:
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
	  // evaluating the BSDF.
	  // Start off with just a big kernel that handles all the different
	  // materials you have in the scenefile.
	  // TODO: compare between directly shading the path segments and shading
	  // path segments that have been reshuffled to be contiguous in memory.

		shadeBSDFMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			remaining_paths,
			dev_intersections,
			dev_paths,
			dev_materials
			);
		//iterationComplete = true;

		if (depth >= traceDepth) {
			iterationComplete = true;
		}

#if STREAM_COMPACTION
		PathSegment* new_dev_path_end = thrust::stable_partition(thrust::device, dev_paths, dev_paths + remaining_paths, is_zero());
		remaining_paths = new_dev_path_end - dev_paths;
		if (remaining_paths <= 0) {
			iterationComplete = true;
		}
#endif // STREAM_COMPACTION
	}

#if PROFILE_ENABLE
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float t;
	cudaEventElapsedTime(&t, start, stop);

	totalTime += t;
	if (countStart && iter > 20) {
		std::cout << totalTime / iter << std::endl;
		countStart = false;
	}
#endif // PROFILE_ENABLE

	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather << <numBlocksPixels, blockSize1d >> > (num_paths, dev_image, dev_paths);
	
	///////////////////////////////////////////////////////////////////////////

	// CHECKITOUT: use dev_image as reference if you want to implement saving denoised images.
	// Otherwise, screenshots are also acceptable.
	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}


// CHECKITOUT: this kernel "post-processes" the gbuffer/gbuffers into something that you can visualize for debugging.
void showGBuffer(uchar4* pbo) {
	const Camera& cam = hst_scene->state.camera;
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
	(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);
	
	// CHECKITOUT: process the gbuffer results and send them to OpenGL buffer for visualization
	gbufferToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, dev_gBuffer);
}

void showImage(uchar4* pbo, int iter) {
	const Camera& cam = hst_scene->state.camera;
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
	(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);
}

void showDenoisedImage(uchar4* pbo) {
	const Camera& cam = hst_scene->state.camera;
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
	(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, 1, dev_denoised_image);
}

// apply the denoise shader
__global__ void applyFilter(
	int stepWidth,
	glm::ivec2 resolution, 
	glm::vec3* i_data, 
	glm::vec3* o_data,
	GBufferPixel* gBuffer, 
	int filter_size, 
	float *kernel,
	float c_phi,
	float n_phi,
	float p_phi
	) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		glm::vec3 sum = glm::vec3(0.0);
		glm::vec3 cval = i_data[index];
		glm::vec3 nval = gBuffer[index].normal;
		glm::vec3 pval = gBuffer[index].pos;

		float cum_w = 0.0;
		int halfFilterWidth = filter_size / 2;
		for (int i = -halfFilterWidth; i <= halfFilterWidth; i++) {
			for (int j = -halfFilterWidth; j <= halfFilterWidth; j++) {
				float xx = x + stepWidth * i;
				float yy = y + stepWidth * j;

				xx = glm::clamp(xx, 0.0f, resolution.x - 1.0f);
				yy = glm::clamp(yy, 0.0f, resolution.y - 1.0f);
				
				int neighboor = xx + (yy * resolution.x);

				glm::vec3 ctmp = i_data[neighboor];
				glm::vec3 t = cval - ctmp;
				float dist2 = glm::dot(t, t);
				float c_w = min(expf(-(dist2) / c_phi), 1.0);

				glm::vec3 ntmp = gBuffer[neighboor].normal;
				t = nval - ntmp;
				dist2 = max(glm::dot(t, t) / (stepWidth * stepWidth), 0.0);
				float n_w = min(expf(-(dist2) / n_phi), 1.0);

				glm::vec3 ptmp = gBuffer[neighboor].pos;
				t = pval - ptmp;
				dist2 = glm::dot(t, t);
				float p_w = min(expf(-(dist2) / p_phi), 1.0);

				float weight = c_w * n_w * p_w;
				int kernelIndex = (i + halfFilterWidth) + (j + halfFilterWidth) * filter_size;
				sum += ctmp * weight * kernel[kernelIndex];
				cum_w += weight * kernel[kernelIndex];
			}
		}
		o_data[index] = sum / cum_w;
	}
}

void denoise(uchar4* pbo, int iter, int filter_size, float cw, float nw, float pw) {
	const Camera& cam = hst_scene->state.camera;
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
	(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	initDenoisedImage << < blocksPerGrid2d, blockSize2d >> > (dev_image, dev_denoised_image, cam.resolution, iter);

	for (int i = 0; i < filter_size; i++) {
		applyFilter << <blocksPerGrid2d, blockSize2d >> > (1 << i, cam.resolution, dev_denoised_image, dev_denoised_image2, dev_gBuffer, 5, dev_kernel, cw, nw, pw);
		std::swap(dev_denoised_image, dev_denoised_image2);
	}
	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_denoised_image);
}

// CHECKITOUT: this kernel "post-processes" the gbuffer/gbuffers into something that you can visualize for debugging.
void showGBuffer(uchar4* pbo) {
    const Camera &cam = hst_scene->state.camera;
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
            (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
            (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // CHECKITOUT: process the gbuffer results and send them to OpenGL buffer for visualization
    gbufferToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, dev_gBuffer);
}

void showImage(uchar4* pbo, int iter) {
const Camera &cam = hst_scene->state.camera;
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
            (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
            (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);
}
