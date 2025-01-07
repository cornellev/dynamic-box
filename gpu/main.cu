#include <iostream>
#include <filesystem>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

#define ASSERT_CUDA(call) assert(call == cudaSuccess)

static const int DEVICE_ID = 0;

static const Eigen::Matrix3f K_l{{699.41, 0.0, 652.8}, {0.0, 699.365, 358.133}, {0.0, 0.0, 1.0}};
static const Eigen::Matrix3f K_r{{697.635, 0.0, 671.665}, {0.0, 697.63, 354.611}, {0.0, 0.0, 1.0}};

static const Eigen::Matrix3f T{
    {0.0, 0.467919, 0.0458908}, {-0.467919, 0.0, -119.905}, {-0.0458908, 119.905, 0.0}};
static const Eigen::Vector3f R_Rodrigues = Eigen::Vector3f{0.00239722, 0.00697667, -0.0021326};
static const Eigen::Matrix3f R =
    Eigen::AngleAxisf(R_Rodrigues.norm(), R_Rodrigues.normalized()).toRotationMatrix();

static const Eigen::Matrix3f E = R * T;
static const Eigen::Matrix3f F = K_r.transpose().inverse() * E * K_l.inverse();

__device__ static const int32_t SAD_KERNEL_SIZE = 11;
static_assert(SAD_KERNEL_SIZE % 2 == 1, "SAD kernel size must be odd");

__device__ static const int32_t MIN_DISPARITY = 1;    // inclusive
__device__ static const int32_t MAX_DISPARITY = 128;  // inclusive
static_assert(MIN_DISPARITY >= 0, "Min disparity must be at least 0");
static_assert(MAX_DISPARITY > MIN_DISPARITY, "Max disparity must be greater than min disparity");
static_assert((MAX_DISPARITY - MIN_DISPARITY + 1) % 16 == 0,
    "Num disparities must be divisble by 16 for OpenCV");

static const float MIN_DISPLAY_DEPTH = 0.0;
static const float MAX_DISPLAY_DEPTH = 10000.0;

__global__ void calculateDepth(const uchar* left, const uchar* right, float* depth, int32_t rows,
    int32_t cols, Eigen::Matrix3f F) {
    int32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    int32_t pixel_row = thread_id / cols;
    int32_t pixel_col = thread_id % cols;

    if (pixel_row >= rows || pixel_col >= cols) {
        return;
    }

    Eigen::Vector3f pixel_homogenous{pixel_col, pixel_row, 1.0};
    Eigen::Vector3f epipolar_line = F * pixel_homogenous;
    int32_t epipolar_r_row = std::round(-epipolar_line(2) / epipolar_line(1));
    if (epipolar_r_row < 0 || epipolar_r_row >= rows) {
        return;
    }

    // inclusive
    int32_t block_start = std::clamp(pixel_col - MAX_DISPARITY - SAD_KERNEL_SIZE / 2, 0, cols);
    // exclusive
    int32_t block_end = std::clamp(pixel_col - MIN_DISPARITY + SAD_KERNEL_SIZE / 2 + 1, 0, cols);

    int32_t max_disparity_here = std::min(pixel_col, MAX_DISPARITY);

    // [0] -> MIN_DISPARITY
    // [len(sads) - 1] -> MAX_DISPARITY
    uint16_t sads[MAX_DISPARITY - MIN_DISPARITY];

    for (int32_t ker_row = 0; ker_row < SAD_KERNEL_SIZE; ker_row++) {
        int32_t l_row = pixel_row + ker_row - SAD_KERNEL_SIZE / 2;
        int32_t r_row = epipolar_r_row + ker_row - SAD_KERNEL_SIZE / 2;

        if (l_row < 0 || l_row >= rows || r_row < 0 || r_row >= rows) continue;

        for (int32_t r_col = block_start; r_col < block_end; r_col++) {
            int32_t r = right[r_row * cols + r_col];

            for (int32_t ker_col = 0; ker_col < SAD_KERNEL_SIZE / 2; ker_col++) {
                int32_t corresponding_center = r_col + SAD_KERNEL_SIZE / 2 + ker_col;
                int32_t corresponding_disparity = pixel_col - corresponding_center;

                if (corresponding_disparity >= MIN_DISPARITY
                    && corresponding_disparity <= max_disparity_here) {
                    int32_t l_col = pixel_col + ker_col - SAD_KERNEL_SIZE / 2;
                    int32_t l = left[l_row * cols + l_col];

                    int32_t sad_idx = corresponding_disparity - MIN_DISPARITY;
                    sads[sad_idx] += std::abs(r - l);
                }
            }
        }
    }

    uint16_t min_sad = UINT16_MAX;
    int32_t best_disp = MIN_DISPARITY;

    for (int32_t disp = MIN_DISPARITY; disp <= max_disparity_here; disp++) {
        if (sads[disp] < min_sad) {
            min_sad = sads[disp];
            best_disp = disp;
        }
    }

    depth[thread_id] = 119.905 * 699.41 / best_disp;
}

int main(int argc, char** argv) {
    ASSERT_CUDA(cudaSetDevice(DEVICE_ID));
    ASSERT_CUDA(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

    cv::Mat3b left = cv::imread("../left/left1.png", cv::IMREAD_COLOR);
    cv::Mat3b right = cv::imread("../right/right1.png", cv::IMREAD_COLOR);
    assert(left.rows == right.rows && left.cols == right.cols);

    cv::Mat1b left_gray;
    cv::cvtColor(left, left_gray, cv::COLOR_BGR2GRAY);
    cv::Mat1b right_gray;
    cv::cvtColor(right, right_gray, cv::COLOR_BGR2GRAY);

    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat1f depth(left_gray.rows, left_gray.cols);

    size_t stereo_bytes = left_gray.total() * left_gray.elemSize();
    size_t depth_bytes = depth.total() * depth.elemSize();

    uchar* d_left;
    uchar* d_right;
    float* d_depth;
    ASSERT_CUDA(cudaMalloc(&d_left, stereo_bytes));
    ASSERT_CUDA(cudaMalloc(&d_right, stereo_bytes));
    ASSERT_CUDA(cudaMalloc(&d_depth, depth_bytes));

    ASSERT_CUDA(cudaMemcpy(d_left, left_gray.data, stereo_bytes, cudaMemcpyHostToDevice));
    ASSERT_CUDA(cudaMemcpy(d_right, right_gray.data, stereo_bytes, cudaMemcpyHostToDevice));

    size_t threads = left_gray.rows * left_gray.cols;
    size_t threads_per_block = 128;
    size_t blocks_per_grid = (threads + threads_per_block - 1) / threads_per_block;

    std::cout << "Using " << threads_per_block << " threads per block and " << blocks_per_grid
              << " blocks\n";

    calculateDepth<<<blocks_per_grid, threads_per_block>>>(d_left, d_right, d_depth, left_gray.rows,
        left_gray.cols, F);

    ASSERT_CUDA(cudaDeviceSynchronize());

    ASSERT_CUDA(cudaMemcpy(depth.data, d_depth, depth_bytes, cudaMemcpyDeviceToHost));

    ASSERT_CUDA(cudaFree(d_left));
    ASSERT_CUDA(cudaFree(d_right));
    ASSERT_CUDA(cudaFree(d_depth));

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Depth calculation took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms"
              << std::endl;

    auto start_cv = std::chrono::high_resolution_clock::now();

    cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create();
    stereo->setBlockSize(SAD_KERNEL_SIZE);
    stereo->setMinDisparity(MIN_DISPARITY);
    stereo->setNumDisparities(MAX_DISPARITY - MIN_DISPARITY + 1);

    cv::Mat disparity_cv;
    stereo->compute(left_gray, right_gray, disparity_cv);
    cv::Mat1f depth_cv;
    disparity_cv.convertTo(depth_cv, CV_32F);
    depth_cv = 119.905 * 699.41 / (depth_cv / 16.0);

    auto end_cv = std::chrono::high_resolution_clock::now();
    std::cout << "OpenCV depth calculation took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_cv - start_cv).count()
              << " ms" << std::endl;

    cv::Mat1b depth_8u;
    depth.convertTo(depth_8u, CV_8U, 255.0 / (MAX_DISPLAY_DEPTH - MIN_DISPLAY_DEPTH),
        -255.0 * MIN_DISPLAY_DEPTH / (MAX_DISPLAY_DEPTH - MIN_DISPLAY_DEPTH));

    cv::Mat1b depth_cv_8u;
    depth_cv.convertTo(depth_cv_8u, CV_8U, 255.0 / (MAX_DISPLAY_DEPTH - MIN_DISPLAY_DEPTH),
        -255.0 * MIN_DISPLAY_DEPTH / (MAX_DISPLAY_DEPTH - MIN_DISPLAY_DEPTH));

    std::filesystem::create_directory("output");
    cv::imwrite("output/depth.png", depth_8u);
    cv::imwrite("output/depth_cv.png", depth_cv_8u);

    if (argc > 1 && std::string(argv[1]) == "display") {
        cv::imshow("Depth", depth_8u);
        cv::waitKey(0);
    }

    return 0;
}