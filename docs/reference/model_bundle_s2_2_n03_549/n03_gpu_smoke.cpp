// #549 S2-2 N03 research GPU smoke (evidence only, not production).
//
// usage: n03_gpu_smoke <path|proposed> <op_path> <expected_sha256>
//
// Loads N03 either by path after a pre-load hash (main@74f2f5cb) or through
// the proposal (one source descriptor, verified buffer, labeled sealed memfd,
// readback hash, dlopen /proc/self/fd/<M>), then calls the registered native
// op saccade_native::selective_scan_fwd on cuda:0 eagerly (float32 and
// float16, per-channel A + D) and under a CUDA graph capture + 3 replays
// (float32). Prints SHA-256 of each output's bytes; the graph replays must
// equal the eager output bit for bit. Exit 0 ok, 2 host refusal, 3 mismatch.
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <link.h>
#include <openssl/evp.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <torch/torch.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

std::string sha_hex(const void* p, size_t n) {
    unsigned char d[32];
    unsigned int dl = 0;
    EVP_Digest(p, n, d, &dl, EVP_sha256(), nullptr);
    char out[65];
    for (int i = 0; i < 32; ++i) std::snprintf(out + 2 * i, 3, "%02x", d[i]);
    return out;
}

[[noreturn]] void refuse(const std::string& what) {
    std::printf("host refused: %s\n", what.c_str());
    std::fflush(stdout);
    _exit(2);
}

std::vector<unsigned char> read_all(int fd) {
    struct stat st{};
    if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) refuse("not a regular file");
    std::vector<unsigned char> b(static_cast<size_t>(st.st_size));
    size_t got = 0;
    while (got < b.size()) {
        ssize_t k = pread(fd, b.data() + got, b.size() - got, static_cast<off_t>(got));
        if (k < 0 && errno == EINTR) continue;
        if (k <= 0) refuse("short read");
        got += static_cast<size_t>(k);
    }
    char extra;
    if (pread(fd, &extra, 1, static_cast<off_t>(got)) != 0) refuse("file grew while read");
    return b;
}

std::string tensor_sha(const at::Tensor& t) {
    at::Tensor c = t.contiguous().cpu();
    return sha_hex(c.data_ptr(), c.nbytes());
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) return 64;
    const std::string mode = argv[1], op = argv[2], expected = argv[3];
    std::string load = op;
    int s = -1, m = -1;
    if (mode == "path") {
        int fd = open(op.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) refuse("open");
        auto b = read_all(fd);
        close(fd);
        if (sha_hex(b.data(), b.size()) != expected) refuse("sha256 mismatch");
    } else if (mode == "proposed") {
        s = open(op.c_str(), O_RDONLY | O_CLOEXEC);
        if (s < 0) refuse("open");
        auto b = read_all(s);
        if (sha_hex(b.data(), b.size()) != expected) refuse("sha256 mismatch");
        const std::string label = "saccade-n03:src-fd=" + std::to_string(s);
        m = memfd_create(label.c_str(), MFD_CLOEXEC | MFD_ALLOW_SEALING | MFD_NOEXEC_SEAL);
        if (m < 0 && errno == EINVAL) m = memfd_create(label.c_str(), MFD_CLOEXEC | MFD_ALLOW_SEALING);
        if (m < 0) refuse("memfd_create");
        for (size_t sent = 0; sent < b.size();) {
            ssize_t k = write(m, b.data() + sent, b.size() - sent);
            if (k < 0 && errno == EINTR) continue;
            if (k <= 0) refuse("memfd write");
            sent += static_cast<size_t>(k);
        }
        const int need = F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL;
        if (fcntl(m, F_ADD_SEALS, need) != 0 || (fcntl(m, F_GET_SEALS) & need) != need) refuse("seal");
        auto back = read_all(m);
        if (back.size() != b.size() || sha_hex(back.data(), back.size()) != expected) refuse("memfd readback");
        load = "/proc/self/fd/" + std::to_string(m);
    } else {
        return 64;
    }
    void* h = dlopen(load.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (h == nullptr) refuse(std::string("dlopen: ") + dlerror());
    if (s >= 0) close(s);
    if (m >= 0) close(m);
    struct link_map* lm = nullptr;
    dlinfo(h, RTLD_DI_LINKMAP, &lm);
    std::printf("mode=%s loaded l_name=%s\n", mode.c_str(), lm ? lm->l_name : "?");

    auto op_handle = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("saccade_native::selective_scan_fwd", "")
                         .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                                           const at::Tensor&, const at::Tensor&, int64_t, bool)>();
    torch::manual_seed(549);
    const int64_t Bn = 1, L = 64, Dd = 128, N = 16;
    auto cpu = torch::TensorOptions().dtype(torch::kFloat32);
    at::Tensor u = torch::randn({Bn, L, Dd}, cpu), delta = torch::rand({Bn, L, Dd}, cpu) * 0.1,
               A = -torch::rand({Dd, N}, cpu), B = torch::randn({Bn, L, N}, cpu), C = torch::randn({Bn, L, N}, cpu),
               D = torch::randn({Dd}, cpu);
    auto cuda = [](const at::Tensor& t) { return t.to(torch::kCUDA); };
    at::Tensor gu = cuda(u), gd = cuda(delta), gA = cuda(A), gB = cuda(B), gC = cuda(C), gD = cuda(D);

    at::Tensor y32 = op_handle.call(gu, gd, gA, gB, gC, gD, 1, false);
    auto h16 = [](const at::Tensor& t) { return t.to(torch::kHalf); };
    at::Tensor y16 = op_handle.call(h16(gu), h16(gd), h16(gA), h16(gB), h16(gC), h16(gD), 1, true);
    torch::cuda::synchronize();
    const std::string e32 = tensor_sha(y32), e16 = tensor_sha(y16);
    std::printf("eager_f32_sha256=%s\neager_f16_sha256=%s\n", e32.c_str(), e16.c_str());

    // CUDA graph capture of the op on a side stream, then 3 replays.
    c10::cuda::CUDAStream side = c10::cuda::getStreamFromPool();
    at::Tensor static_y;
    at::cuda::CUDAGraph graph;
    {
        c10::cuda::CUDAStreamGuard g(side);
        for (int i = 0; i < 3; ++i) static_y = op_handle.call(gu, gd, gA, gB, gC, gD, 1, false);  // warm-up
        side.synchronize();
        graph.capture_begin();
        static_y = op_handle.call(gu, gd, gA, gB, gC, gD, 1, false);
        graph.capture_end();
    }
    int bad = 0;
    for (int r = 0; r < 3; ++r) {
        static_y.zero_();
        graph.replay();
        torch::cuda::synchronize();
        const std::string rs = tensor_sha(static_y);
        std::printf("graph_replay_%d_sha256=%s equal_to_eager=%d\n", r, rs.c_str(), rs == e32);
        bad += rs != e32;
    }
    std::printf("graph_replays_equal=%d\n", bad == 0);
    std::fflush(stdout);
    _exit(bad == 0 ? 0 : 3);
}
