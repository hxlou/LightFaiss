#include "src/backend/gpu-kompute/L2Norm.hpp"
#include "src/backend/gpu-kompute/distance.hpp"
#include "src/backend/gpu-kompute/readShader.hpp"

#include <kompute/Kompute.hpp>

namespace gpu_kompute {

void normalized_L2(
    kp::Manager* mgr,
    size_t dim,
    size_t nx,
    float* x
) {
    fvec_renorm_L2(mgr, dim, nx, x);
}

void fvec_renorm_L2(
    kp::Manager* mgr,
    size_t dim,         // 向量的维度
    size_t nx,          // 向量的数量
    float* x            // 输入向量，nx * dim
) {
    // 与vecsNorm类似，但是这个函数是原地进行修改，而不是进行额外的计算
    auto shader = readSpvFile("src/backend/gpu-kompute/shaders/L2ReNorm.comp.spv");
    std::shared_ptr<kp::TensorT<float>> vecs = mgr->tensorT<float>(std::vector<float>(x, x + nx * dim));

    std::vector<uint32_t> pushConsts = {
        static_cast<uint32_t>(nx),
        static_cast<uint32_t>(dim)
    };

    std::vector<std::shared_ptr<kp::Memory>> memories = {
        std::static_pointer_cast<kp::Memory>(vecs)
    };

    auto algorithm = mgr->algorithm(
        memories,
        shader,
        kp::Workgroup({static_cast<uint32_t>(nx), 1, 1}),
        std::vector<float>{},
        pushConsts
    );

    auto seq = mgr->sequence()
        ->record<kp::OpSyncDevice>(memories)
        ->record<kp::OpAlgoDispatch>(algorithm)
        ->record<kp::OpSyncLocal>({vecs});

    seq->eval();

    // 把数据从vecs拷贝到x中
    memcpy(x, vecs->data(), nx * dim * sizeof(float));
}

}