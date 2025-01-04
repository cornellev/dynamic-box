#include <iostream>
#include <filesystem>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

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

static const int32_t SAD_KERNEL_SIZE = 11;
static_assert(SAD_KERNEL_SIZE % 2 == 1, "SAD kernel size must be odd");

static const int32_t MIN_DISPARITY = 0;
static const int32_t MAX_DISPARITY = 128;

static const float MIN_DISPLAY_DEPTH = 0.0;
static const float MAX_DISPLAY_DEPTH = 10000.0;

__global__ void calculateDepth(const uchar3* left, const uchar3* right, float* depth, int32_t rows,
    int32_t cols, Eigen::Matrix3f F) {
    extern __shared__ uchar3 shared[];

    int32_t id = blockIdx.x * blockDim.x + threadIdx.x;

    int32_t row = id / cols;
    int32_t col = id % cols;

    for (int32_t i = 0; i < SAD_KERNEL_SIZE; i++) {
        int32_t s_idx = blockDim.x * i + threadIdx.x;
        int32_t row_offset = i - SAD_KERNEL_SIZE / 2;

        int32_t r_row = (int32_t)row + row_offset;
        if (r_row < 0 || r_row >= rows) {
            shared[s_idx] = {0, 0, 0};
        } else {
            shared[s_idx] = right[r_row * cols + col];
        }
    }

    __syncthreads();

    Eigen::Vector3f l_pixel{col, row, 1.0};
    Eigen::Vector3f epipolar_line = F * l_pixel;
    int32_t r_row = round(-epipolar_line(2) / epipolar_line(1));

    int32_t search_start = max(0, col - MAX_DISPARITY);
    int32_t search_end = max(0, min(col, col - MIN_DISPARITY));

    uint32_t min_sad = UINT32_MAX;
    int32_t best_r_col = search_start;

    uint64_t shared_count = 0;
    uint64_t global_count = 0;

    for (int32_t r_col = search_start; r_col <= search_end; r_col++) {
        uint32_t sad = 0;

        for (int32_t i = 0; i < SAD_KERNEL_SIZE; i++) {
            for (int32_t j = 0; j < SAD_KERNEL_SIZE; j++) {
                int32_t l_ker_row = row + i - SAD_KERNEL_SIZE / 2;
                int32_t l_ker_col = col + j - SAD_KERNEL_SIZE / 2;
                int32_t r_ker_row = r_row + i - SAD_KERNEL_SIZE / 2;
                int32_t r_ker_col = r_col + j - SAD_KERNEL_SIZE / 2;

                if (l_ker_row < 0 || l_ker_row >= rows || l_ker_col < 0 || l_ker_col >= cols
                    || r_ker_row < 0 || r_ker_row >= rows || r_ker_col < 0 || r_ker_col >= cols) {
                    continue;
                }

                int32_t s_r_i = i;
                int32_t s_r_j = threadIdx.x - (col - r_ker_col);
                int32_t s_idx = blockDim.x * s_r_i + s_r_j;

                uchar3 l_pixel = left[l_ker_row * cols + l_ker_col];
                uchar3 r_pixel;
                if (s_r_j >= 0 && s_r_j < blockDim.x) {
                    r_pixel = shared[s_idx];
                    shared_count++;
                } else {
                    r_pixel = right[r_ker_row * cols + r_ker_col];
                    global_count++;
                }

                sad += abs(l_pixel.x - r_pixel.x) + abs(l_pixel.y - r_pixel.y)
                       + abs(l_pixel.z - r_pixel.z);
            }
        }

        // TODO: branchless?
        if (sad < min_sad) {
            min_sad = sad;
            best_r_col = r_col;
        }
    }

    float disparity = col - best_r_col;
    depth[id] = 119.905 * 699.41 / disparity;  // mm
}

int main() {
    cudaSetDevice(DEVICE_ID);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

    cv::Mat3b left = cv::imread("../left/left1.png", cv::IMREAD_COLOR);
    cv::Mat3b right = cv::imread("../right/right1.png", cv::IMREAD_COLOR);
    assert(left.rows == right.rows && left.cols == right.cols);

    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat1f depth(left.rows, left.cols);

    int32_t stereo_bytes = left.total() * left.elemSize();
    int32_t depth_bytes = depth.total() * depth.elemSize();

    uchar3* d_left;
    uchar3* d_right;
    float* d_depth;
    cudaMalloc(&d_left, stereo_bytes);
    cudaMalloc(&d_right, stereo_bytes);
    cudaMalloc(&d_depth, depth_bytes);

    cudaMemcpy(d_left, left.data, stereo_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_right, right.data, stereo_bytes, cudaMemcpyHostToDevice);

    int32_t threads = left.rows * left.cols;
    int32_t threads_per_block = 1024;
    int32_t blocks_per_grid = (threads + threads_per_block - 1) / threads_per_block;
    int32_t shared_mem = threads_per_block * SAD_KERNEL_SIZE * sizeof(uchar3);

    std::cout << "Using " << threads_per_block << " threads per block and " << blocks_per_grid
              << " blocks\n";
    std::cout << "Using " << shared_mem / 1024.0 << "kiB of shared memory per block\n";

    auto start_kernel = std::chrono::high_resolution_clock::now();
    calculateDepth<<<blocks_per_grid, threads_per_block, shared_mem>>>(d_left, d_right, d_depth,
        left.rows, left.cols, F);

    cudaDeviceSynchronize();
    auto end_kernel = std::chrono::high_resolution_clock::now();
    std::cout
        << "Kernel execution took "
        << std::chrono::duration_cast<std::chrono::milliseconds>(end_kernel - start_kernel).count()
        << " ms" << std::endl;

    cudaMemcpy(depth.data, d_depth, depth_bytes, cudaMemcpyDeviceToHost);

    cv::Mat1b depth_8u;
    depth.convertTo(depth_8u, CV_8U, 255.0 / (MAX_DISPLAY_DEPTH - MIN_DISPLAY_DEPTH),
        -255.0 * MIN_DISPLAY_DEPTH / (MAX_DISPLAY_DEPTH - MIN_DISPLAY_DEPTH));

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Depth calculation took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms"
              << std::endl;

    start = std::chrono::high_resolution_clock::now();

    cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create();
    stereo->setBlockSize(SAD_KERNEL_SIZE);
    stereo->setMinDisparity(MIN_DISPARITY);
    stereo->setNumDisparities(MAX_DISPARITY - MIN_DISPARITY);

    cv::Mat1b left_gray;
    cv::cvtColor(left, left_gray, cv::COLOR_BGR2GRAY);
    cv::Mat1b right_gray;
    cv::cvtColor(right, right_gray, cv::COLOR_BGR2GRAY);

    cv::Mat disparity_cv;
    stereo->compute(left_gray, right_gray, disparity_cv);
    cv::Mat1f depth_cv;
    disparity_cv.convertTo(depth_cv, CV_32F);
    depth_cv = 119.905 * 699.41 / (depth_cv / 16.0);
    cv::Mat1b depth_cv_8u;
    depth_cv.convertTo(depth_cv_8u, CV_8U, 255.0 / (MAX_DISPLAY_DEPTH - MIN_DISPLAY_DEPTH),
        -255.0 * MIN_DISPLAY_DEPTH / (MAX_DISPLAY_DEPTH - MIN_DISPLAY_DEPTH));

    end = std::chrono::high_resolution_clock::now();
    std::cout << "OpenCV depth calculation took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms"
              << std::endl;

    std::filesystem::create_directory("output");
    cv::imwrite("output/depth.png", depth_8u);
    cv::imwrite("output/depth_cv.png", depth_cv_8u);

    cv::imshow("Depth", depth_8u);
    cv::waitKey(0);

    cudaFree(d_left);
    cudaFree(d_right);
    cudaFree(d_depth);

    return 0;
}