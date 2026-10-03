// Native ingest plan and sequence input (#465 Phase B PR-7).
// See saccade_shipping/ingest_plan.hpp.
#include "saccade_shipping/ingest_plan.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <sstream>
#include <system_error>

namespace saccade::shipping {
namespace {

void require(bool ok, const std::string& what) {
    if (!ok) {
        throw ConfigError("shipping ingest: " + what +
                          " (the U3b-1 ingest has no implementation for it)");
    }
}

[[noreturn]] void bad_input(const std::string& what) {
    throw InputError("shipping ingest: " + what);
}

std::string strip(const std::string& s) {
    const auto ws = [](unsigned char c) { return std::isspace(c) != 0; };
    auto b = std::find_if_not(s.begin(), s.end(), ws);
    auto e = std::find_if_not(s.rbegin(), s.rend(), ws).base();
    return b < e ? std::string(b, e) : std::string();
}

std::string lower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

bool ascii(const std::string& s) {
    return std::all_of(s.begin(), s.end(),
                       [](char c) { return static_cast<unsigned char>(c) < 0x80; });
}

// configparser getint() on the plain form: [+-]digits.
int parse_int(const std::string& key, const std::string& v) {
    std::size_t i = (!v.empty() && (v[0] == '+' || v[0] == '-')) ? 1 : 0;
    if (i == v.size() || !std::all_of(v.begin() + static_cast<std::ptrdiff_t>(i), v.end(),
                                      [](char c) { return c >= '0' && c <= '9'; })) {
        bad_input("seqinfo.ini " + key + " is not a plain integer: '" + v + "'");
    }
    long long out = 0;
    for (std::size_t k = i; k < v.size(); ++k) {
        out = out * 10 + (v[k] - '0');
        if (out > std::numeric_limits<int>::max()) bad_input("seqinfo.ini " + key + " out of range");
    }
    return static_cast<int>(v[0] == '-' ? -out : out);
}

}  // namespace

IngestPlan plan_ingest(const ResolvedShippingConfig& cfg) {
    const auto& s = cfg.host_params.steps;
    const auto& c = cfg.host_params.cfg;
    // pipeline.py: SACCADE_GPU_DECODE=1 selects TorchvisionGpuStreamer (nvJPEG);
    // otherwise DALIStreamerStream decodes on the CPU, a different decoder.
    require(s.ingest_gpu_decode, "steps.ingest.gpu_decode must be true (nvJPEG decode)");
    // SACCADE_NV12_BUFFER=1 converts the frame to NV12 (and, without
    // preprocess modes, skips the float32 frame buffer entirely).
    require(!s.ingest_nv12_buffer, "steps.ingest.nv12_buffer must be false");
    // apply_frame_preprocess returns at once only for an empty mode list;
    // gamma/contrast rewrite the frame buffer, letterbox changes the detector input.
    require(c.preprocess_modes.empty(), "cfg.preprocess_modes must be empty");
    // The workbench has an ingest of its own (evaluator.py, `.float() / 255.0`).
    require(!s.track_workbench, "steps.track.workbench must be false");
    return IngestPlan{};
}

SeqInfo parse_seqinfo(const std::string& text) {
    std::map<std::string, std::map<std::string, std::string>> sections;
    std::map<std::string, std::string>* current = nullptr;
    std::istringstream in(text);
    std::string raw;
    int line_no = 0;
    while (std::getline(in, raw)) {
        ++line_no;
        if (!raw.empty() && raw.back() == '\r') raw.pop_back();
        const std::string where = "seqinfo.ini line " + std::to_string(line_no);
        const std::string line = strip(raw);
        if (line.empty() || line[0] == '#' || line[0] == ';') continue;
        if (std::isspace(static_cast<unsigned char>(raw[0]))) {
            bad_input(where + ": indented line (configparser continuation) is not supported");
        }
        if (line.front() == '[') {
            if (line.back() != ']' || line.size() < 3) bad_input(where + ": malformed section header");
            const std::string name = line.substr(1, line.size() - 2);
            if (name == "DEFAULT") bad_input(where + ": a DEFAULT section is not supported");
            if (sections.count(name) != 0) bad_input(where + ": duplicate section [" + name + "]");
            current = &sections[name];
            continue;
        }
        if (current == nullptr) bad_input(where + ": key before any section header");
        const std::size_t d = line.find_first_of("=:");
        if (d == std::string::npos) bad_input(where + ": not a key = value line");
        const std::string key = lower(strip(line.substr(0, d)));
        const std::string value = strip(line.substr(d + 1));
        if (key.empty()) bad_input(where + ": empty key");
        if (value.find('%') != std::string::npos) bad_input(where + ": '%' in a value is not supported");
        if (!current->emplace(key, value).second) bad_input(where + ": duplicate key " + key);
    }
    const auto sec = sections.find("Sequence");
    if (sec == sections.end()) bad_input("seqinfo.ini has no [Sequence] section");
    const auto get = [&](const char* key) {
        const auto it = sec->second.find(lower(key));
        if (it == sec->second.end()) bad_input(std::string("seqinfo.ini [Sequence] has no ") + key);
        return parse_int(key, it->second);
    };
    SeqInfo info;
    info.im_width = get("imWidth");
    info.im_height = get("imHeight");
    info.seq_length = get("seqLength");
    if (info.im_width <= 0 || info.im_height <= 0) bad_input("seqinfo.ini imWidth/imHeight must be positive");
    if (info.seq_length < 0) bad_input("seqinfo.ini seqLength must not be negative");
    return info;
}

SequenceInput read_sequence_input(const std::filesystem::path& sequence_dir, int max_frames) {
    namespace fs = std::filesystem;
    const fs::path ini = sequence_dir / "seqinfo.ini";
    std::ifstream f(ini, std::ios::binary);
    if (!f) bad_input("cannot read " + ini.string());
    std::ostringstream text;
    text << f.rdbuf();
    const SeqInfo info = parse_seqinfo(text.str());

    SequenceInput in;
    in.img_dir = sequence_dir / "img1";
    in.im_width = info.im_width;
    in.im_height = info.im_height;
    in.seq_length = info.seq_length;
    std::error_code ec;
    if (fs::is_directory(in.img_dir, ec)) {
        for (fs::directory_iterator it(in.img_dir, ec), end; !ec && it != end; it.increment(ec)) {
            const std::string name = it->path().filename().string();
            if (name.size() >= 4 && name.compare(name.size() - 4, 4, ".jpg") == 0) {
                if (!ascii(name)) bad_input("non-ASCII frame file name in " + in.img_dir.string());
                in.listed.push_back(name);
            }
        }
        if (ec) bad_input("cannot list " + in.img_dir.string() + ": " + ec.message());
    }
    std::sort(in.listed.begin(), in.listed.end());
    const int frame_end = max_frames > 0 ? std::min(max_frames, in.seq_length) : in.seq_length;
    if (static_cast<std::size_t>(frame_end) > in.listed.size()) {
        bad_input(in.img_dir.string() + " lists " + std::to_string(in.listed.size()) +
                  " .jpg entries, fewer than the " + std::to_string(frame_end) +
                  " frames to consume (the oracle would truncate the sequence)");
    }
    in.frames.assign(in.listed.begin(), in.listed.begin() + frame_end);
    return in;
}

std::vector<std::uint8_t> read_frame_file(const std::filesystem::path& path) {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec)) bad_input(path.string() + " is not a regular file");
    std::ifstream f(path, std::ios::binary);
    if (!f) bad_input("cannot read " + path.string());
    std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (f.bad()) bad_input("error reading " + path.string());
    if (bytes.empty()) bad_input(path.string() + " is empty");
    return bytes;
}

}  // namespace saccade::shipping
