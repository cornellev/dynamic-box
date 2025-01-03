#include <iostream>
#include <cuda_runtime.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

static const int DEVICE_ID = 0;
static const size_t SAD_KERNEL_SIZE = 15;

static const Eigen::Matrix3f K_l{{699.41, 0.0, 652.8}, {0.0, 699.365, 358.133}, {0.0, 0.0, 1.0}};
static const Eigen::Matrix3f K_r{{697.635, 0.0, 671.665}, {0.0, 697.63, 354.611}, {0.0, 0.0, 1.0}};

static const Eigen::Matrix3f T{
    {0.0, 0.467919, 0.0458908}, {-0.467919, 0.0, -119.905}, {-0.0458908, 119.905, 0.0}};
static const Eigen::Vector3f R_Rodrigues = Eigen::Vector3f{0.00239722, 0.00697667, -0.0021326};
static const Eigen::Matrix3f R =
    Eigen::AngleAxisf(R_Rodrigues.norm(), R_Rodrigues.normalized()).toRotationMatrix();

static const Eigen::Matrix3f E = R * T;
static const Eigen::Matrix3f F = K_r.transpose().inverse() * E * K_l.inverse();

__device__ uint32_t calculateSAD(const uchar3* left, const uchar3* right, size_t rows, size_t cols,
    size_t l_row, size_t l_col, size_t r_row, size_t r_col) {
    uint32_t sad = 0;

    int32_t lower_neighbors = (SAD_KERNEL_SIZE - 1) / 2;
    int32_t upper_neighbors = SAD_KERNEL_SIZE / 2;

    for (int32_t i = -lower_neighbors; i <= upper_neighbors; i++) {
        for (int32_t j = -lower_neighbors; j <= upper_neighbors; j++) {
            int32_t l_row_idx = l_row + i;
            int32_t l_col_idx = l_col + j;
            int32_t r_row_idx = r_row + i;
            int32_t r_col_idx = r_col + j;

            // TODO: extend image borders
            if (l_row_idx < 0 || l_row_idx >= rows || l_col_idx < 0 || l_col_idx >= cols
                || r_row_idx < 0 || r_row_idx >= rows || r_col_idx < 0 || r_col_idx >= cols) {
                continue;
            }

            uchar3 l_pixel = left[l_row_idx * cols + l_col_idx];
            uchar3 r_pixel = right[r_row_idx * cols + r_col_idx];

            sad += abs(l_pixel.x - r_pixel.x) + abs(l_pixel.y - r_pixel.y)
                   + abs(l_pixel.z - r_pixel.z);
        }
    }

    return sad;
}

__global__ void calculateDepth(const uchar3* left, const uchar3* right, float* depth, size_t rows,
    size_t cols, Eigen::Matrix3f F) {
    for (size_t i = blockDim.x * blockIdx.x + threadIdx.x; i < rows * cols;
        i += blockDim.x * gridDim.x) {
        size_t row = i / cols;
        size_t col = i % cols;

        Eigen::Vector3f l_pixel{col, row, 1.0};
        Eigen::Vector3f epipolar_line = F * l_pixel;
        size_t r_row = round(-epipolar_line(2) / epipolar_line(1));  // is this right?

        uint32_t min_sad = UINT32_MAX;
        size_t best_r_col = 0;
        for (size_t r_col = 0; r_col < col; ++r_col) {
            uint32_t sad = calculateSAD(left, right, rows, cols, row, col, r_row, r_col);

            // TODO: branchless?
            if (sad < min_sad) {
                min_sad = sad;
                best_r_col = r_col;
            }
        }

        int32_t disparity = col - best_r_col;
        depth[i] = 119.905 * 699.41 / disparity;  // mm
    }
}

int main() {
    cudaSetDevice(DEVICE_ID);

    cv::Mat3b left = cv::imread("../left/left1.png", cv::IMREAD_COLOR);
    cv::Mat3b right = cv::imread("../right/right1.png", cv::IMREAD_COLOR);
    assert(left.rows == right.rows && left.cols == right.cols);

    cv::Mat1f depth(left.rows, left.cols);

    size_t stereo_bytes = left.total() * left.elemSize();
    size_t depth_bytes = depth.total() * depth.elemSize();

    uchar3* d_left;
    uchar3* d_right;
    float* d_depth;
    cudaMalloc(&d_left, stereo_bytes);
    cudaMalloc(&d_right, stereo_bytes);
    cudaMalloc(&d_depth, depth_bytes);

    cudaMemcpy(d_left, left.data, stereo_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_right, right.data, stereo_bytes, cudaMemcpyHostToDevice);

    int num_sm;
    cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, DEVICE_ID);

    size_t threads = left.rows * left.cols;
    size_t threads_per_block = 256;
    size_t blocks_per_grid = 32 * num_sm;
    double pixels_per_thread = threads / (double)(blocks_per_grid * threads_per_block);

    std::cout << num_sm << " SMs available, using " << threads_per_block
              << " threads per block and " << blocks_per_grid << " blocks per grid for "
              << pixels_per_thread << " pixels per thread" << std::endl;

    calculateDepth<<<blocks_per_grid, threads_per_block>>>(d_left, d_right, d_depth, left.rows,
        left.cols, F);

    cudaDeviceSynchronize();

    cudaMemcpy(depth.data, d_depth, depth_bytes, cudaMemcpyDeviceToHost);

    const float min_depth = 0.0;
    const float max_depth = 10000.0;  // mm
    cv::Mat1b depth_8u;
    depth.convertTo(depth_8u, CV_8U, 255.0 / (max_depth - min_depth),
        -255.0 * min_depth / (max_depth - min_depth));

    cv::imshow("Left", left);
    cv::imshow("Depth", depth_8u);
    cv::waitKey(0);

    cudaFree(d_left);
    cudaFree(d_right);
    cudaFree(d_depth);

    return 0;
}