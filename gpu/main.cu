#include <iostream>
#include <cuda_runtime.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// __global__ void calculateDepth(const char3* left, const char3* right, float* depth, size_t rows,
//     size_t cols) {
//     int i = blockDim.x * blockIdx.x + threadIdx.x;
// }

__global__ void greyscale(const uchar3* img, uchar* out, size_t rows, size_t cols) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= rows * cols) {
        return;
    }

    uchar3 val = img[i];
    out[i] = ((int)val.x + (int)val.y + (int)val.z) / 3;
}

int main() {
    cv::Mat3b img = cv::imread("../dog.png", cv::IMREAD_COLOR);
    cv::Mat1b img_grey(img.rows, img.cols);

    size_t img_bytes = img.total() * img.elemSize();
    size_t img_grey_bytes = img_grey.total() * img_grey.elemSize();

    uchar3* d_img;
    uchar* d_img_grey;
    cudaMalloc(&d_img, img_bytes);
    cudaMalloc(&d_img_grey, img_grey_bytes);

    cudaMemcpy(d_img, img.data, img_bytes, cudaMemcpyHostToDevice);

    size_t threads = img.rows * img.cols;
    size_t threadsPerBlock = 256;
    size_t blocksPerGrid = (threads + threadsPerBlock - 1) / threadsPerBlock;

    greyscale<<<blocksPerGrid, threadsPerBlock>>>(d_img, d_img_grey, img.rows, img.cols);

    cudaDeviceSynchronize();

    cudaMemcpy(img_grey.data, d_img_grey, img_grey_bytes, cudaMemcpyDeviceToHost);

    cv::imshow("Greyscale Image", img_grey);
    cv::waitKey(0);

    cudaFree(d_img);
    cudaFree(d_img_grey);

    return 0;
}