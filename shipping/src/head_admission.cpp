// TorchScript head admission (#549 MB-41, F-1). See
// saccade_shipping/head_admission.hpp. The LibTorch behavior relied on is
// v2.11.0's torch/csrc/jit/serialization/{import.cpp, unpickler.cpp,
// import_source.cpp, import_export_helpers.cpp} and
// passes/constant_propagation.cpp.
#include "saccade_shipping/head_admission.hpp"

#include <caffe2/serialize/inline_container.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/api/compilation_unit.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/frontend/lexer.h>
#include <torch/csrc/jit/frontend/parser.h>
#include <torch/csrc/jit/frontend/source_range.h>
#include <torch/csrc/jit/frontend/tree_views.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/mobile/file_format.h>
#include <torch/custom_class.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <utility>

#include "saccade_shipping/model_compatibility.hpp"
#include "saccade_shipping/verified_model_bytes.hpp"

namespace saccade::shipping {

const char* const kHeadAdmissionLibTorch = "2.11.0+cu130";

// N02's operators (mamba_head_s_v14replica_t3_t1_fp32_torchscript.pt,
// sha256 1663ec97...), reviewed: tensor arithmetic, shape and view operations,
// the native scan, and the TorchScript plumbing for constants, attributes,
// lists and tuples. None reads a file or any other external state.
const char* const kHeadAdmissionOperators[] = {
    "aten::Int.Tensor",
    "aten::_convolution",
    "aten::add.Tensor",
    "aten::cat",
    "aten::contiguous",
    "aten::exp",
    "aten::expand",
    "aten::flatten.using_ints",
    "aten::flip",
    "aten::linear",
    "aten::mean.dim",
    "aten::mul.Tensor",
    "aten::neg",
    "aten::pixel_shuffle",
    "aten::reshape",
    "aten::select.int",
    "aten::silu",
    "aten::size.int",
    "aten::slice.Tensor",
    "aten::stack",
    "aten::to.dtype",
    "aten::transpose.int",
    "prim::Constant",
    "prim::GetAttr",
    "prim::ListConstruct",
    "prim::NumToTensor.Scalar",
    "prim::TupleConstruct",
    "saccade_native::selective_scan_fwd",
};
const std::size_t kHeadAdmissionOperatorCount = sizeof kHeadAdmissionOperators / sizeof kHeadAdmissionOperators[0];

namespace {

[[noreturn]] void refuse(const std::string& what) {
    throw DetectorLoadError("shipping detector load: head admission (F-1): " + what);
}

void require_reviewed_libtorch(const std::string& libtorch_build) {
    if (libtorch_build != kHeadAdmissionLibTorch) {
        refuse("the operator list was reviewed for LibTorch " + std::string(kHeadAdmissionLibTorch) +
               ", this runtime is built against " + libtorch_build);
    }
}

// ── step 2: record names ─────────────────────────────────────────────────
bool all_digits(const std::string& s) {
    return !s.empty() && std::all_of(s.begin(), s.end(), [](char c) { return c >= '0' && c <= '9'; });
}

bool identifier_path(const std::string& s) {
    // Segments of [A-Za-z0-9_]+ separated by '/'.
    if (s.empty() || s.front() == '/' || s.back() == '/') return false;
    char prev = '/';
    for (char c : s) {
        const bool ident = c == '_' || (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
        if (!ident && !(c == '/' && prev != '/')) return false;
        prev = c;
    }
    return true;
}

bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

enum class RecordKind { Code, DebugRanges, Pickle, Tensor, NeverRead };

// The record's kind, or refuse. `code/<path>.py` must spell a qualifier
// (qualifierToArchivePath), so the code a class name resolves to is the record
// checked here.
RecordKind record_kind(const std::string& name) {
    if (name == "data.pkl" || name == "constants.pkl") return RecordKind::Pickle;
    if (name.rfind("data/", 0) == 0 && all_digits(name.substr(5))) return RecordKind::Tensor;
    if (name.rfind("constants/", 0) == 0 && all_digits(name.substr(10))) return RecordKind::Tensor;
    if (name.rfind("code/", 0) == 0) {
        if (ends_with(name, ".py") && identifier_path(name.substr(5, name.size() - 5 - 3))) return RecordKind::Code;
        if (ends_with(name, ".py.debug_pkl") && identifier_path(name.substr(5, name.size() - 5 - 13))) {
            return RecordKind::DebugRanges;
        }
    }
    // Read only for the archive version / byte order / log metadata, or
    // never (traced inputs: torch::jit::load passes restore_shapes false).
    if (name == "version" || name == ".data/version" || name == "byteorder" || name == ".data/serialization_id" ||
        name == "traced_inputs.pkl") {
        return RecordKind::NeverRead;
    }
    if (name.rfind("traced_inputs/", 0) == 0 && all_digits(name.substr(14))) return RecordKind::NeverRead;
    refuse("archive record " + name + " is not one the head loader reads");
}

std::string record_text(caffe2::serialize::PyTorchStreamReader& reader, const std::string& name) {
    auto [data, size] = reader.getRecord(name);
    return std::string(static_cast<const char*>(data.get()), size);
}

// ── step 3: code ─────────────────────────────────────────────────────────
// The importer also compiles and runs code while it imports: every parameter
// default, every class constant (`name : Final[T] = expr`) and every NamedTuple
// field default goes through ScriptTypeParser::evaluateDefaults, which runs the
// expression (script_type_parser.cpp). N02 has none of these, so any is refused.
void refuse_parameter_defaults(const torch::jit::TreeRef& tree, const std::string& name) {
    if (tree->kind() == torch::jit::TK_PARAM && torch::jit::Param(tree).defaultValue().present()) {
        refuse("code record " + name + " gives parameter " + torch::jit::Param(tree).ident().name() +
               " a default value, which the loader evaluates");
    }
    for (const torch::jit::TreeRef& child : tree->trees()) refuse_parameter_defaults(child, name);
}

// A class body may hold only what the importer reads without evaluating it:
// methods, attribute annotations (`name : T`), __parameters__ / __buffers__
// (lists of names), and __annotations__["n"] = T (a type).
void admit_class_body(const torch::jit::ClassDef& cls, const std::string& cls_name) {
    using namespace torch::jit;
    // These body rules describe the Module importer only. NamedTuple has a
    // different default evaluator, including for fields named __parameters__
    // or __buffers__. N02 needs only Module classes, so every other class
    // kind is refused before applying any Module metadata exception.
    if (!cls.superclass().present() || cls.superclass().get().kind() != TK_VAR ||
        Var(cls.superclass().get()).name().name() != "Module") {
        refuse("class " + cls_name + " is not a Module class; other class importers are not admitted");
    }
    for (const Stmt& stmt : cls.body()) {
        if (stmt.kind() == TK_DEF) {
            if (Def(stmt).name().name() == "__setstate__") {
                refuse("class " + cls_name + " defines __setstate__, which the loader runs");
            }
            continue;
        }
        if (stmt.kind() != TK_ASSIGN) refuse("class " + cls_name + " has a " + kindToString(stmt.kind()) + " statement");
        const Assign assign(stmt);
        if (assign.lhs().kind() == TK_SUBSCRIPT) {
            const Subscript lhs(assign.lhs());
            if (lhs.value().kind() == TK_VAR && Var(lhs.value()).name().name() == "__annotations__") continue;
            refuse("class " + cls_name + " assigns to a subscript other than __annotations__");
        }
        if (assign.lhs().kind() != TK_VAR) refuse("class " + cls_name + " has an assignment the head loader does not read");
        const std::string attr = Var(assign.lhs()).name().name();
        if (!assign.rhs().present() || attr == "__parameters__" || attr == "__buffers__") continue;
        refuse("class " + cls_name + " assigns " + attr + " a value, which the loader evaluates");
    }
}

// The importer's file grammar (import_source.cpp parseSourceIfNeeded):
// an optional `op_version_set = N` line, then imports, classes and functions.
// Returns the qualified names of the classes the record defines.
std::vector<std::string> admit_code(const std::string& name, const std::string& text) {
    using namespace torch::jit;
    const std::string path = name.substr(5, name.size() - 5 - 3);
    std::string qualifier = path;
    std::replace(qualifier.begin(), qualifier.end(), '/', '.');
    std::vector<std::string> classes;
    try {
        Parser p(std::make_shared<Source>(text, name, 1));
        Lexer& L = p.lexer();
        if (L.cur().kind == TK_IDENT && L.cur().text() == "op_version_set") {
            L.next();
            L.expect('=');
            L.expect(TK_NUMBER);
            L.expect(TK_NEWLINE);
        }
        while (L.cur().kind != TK_EOF) {
            while (L.nextIf(TK_IMPORT)) {
                while (L.cur().kind != TK_NEWLINE) L.next();
                L.expect(TK_NEWLINE);
            }
            if (L.cur().kind == TK_CLASS_DEF) {
                const ClassDef cls(p.parseClass());
                const std::string cls_name = qualifier + "." + cls.name().name();
                admit_class_body(cls, cls_name);
                refuse_parameter_defaults(cls, name);
                classes.push_back(cls_name);
            } else if (L.cur().kind == TK_DEF) {
                const Def fn(p.parseFunction(/*is_method=*/false));
                if (fn.name().name() == "__setstate__") {
                    refuse("code record " + name + " defines a function __setstate__");
                }
                refuse_parameter_defaults(fn, name);
            } else {
                refuse("code record " + name + " has " + kindToString(L.cur().kind) + " at top level");
            }
        }
    } catch (const DetectorLoadError&) {
        throw;
    } catch (const std::exception& e) {
        refuse("code record " + name + " does not parse: " + std::string(e.what()).substr(0, 200));
    }
    return classes;
}

// ── step 4: pickles ──────────────────────────────────────────────────────
struct PickleGlobal {
    std::string module, name;
};

bool python_id_char(char c) {
    // unpickler.cpp is_valid_python_id_char
    return c == '_' || c == '.' || (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

// The GLOBALs of one pickle, read the way LibTorch's Unpickler reads it
// (unpickler.cpp run, readInstruction): PROTO 2, then opcodes up to STOP.
// Any other opcode, a truncated argument or a GLOBAL that is not two
// identifiers is a refusal, so nothing the unpickler would read is skipped.
std::vector<PickleGlobal> pickle_globals(const std::string& record, const std::string& bytes) {
    std::size_t pos = 0;
    const std::size_t n = bytes.size();
    auto bad = [&](const std::string& why) -> void {
        refuse("pickle " + record + " at byte " + std::to_string(pos) + ": " + why);
    };
    auto take = [&](std::size_t k) {
        if (k > n - pos) bad("truncated");
        const std::size_t at = pos;
        pos += k;
        return at;
    };
    auto le = [&](std::size_t k) {
        const std::size_t at = take(k);
        std::uint64_t v = 0;
        for (std::size_t i = 0; i < k; ++i) v |= std::uint64_t(static_cast<unsigned char>(bytes[at + i])) << (8 * i);
        return v;
    };
    auto line = [&]() {
        const std::size_t nl = bytes.find('\n', pos);
        if (nl == std::string::npos) bad("GLOBAL without a newline");
        std::string s = bytes.substr(pos, nl - pos);
        if (s.empty() || !std::all_of(s.begin(), s.end(), python_id_char)) bad("GLOBAL is not a qualified identifier");
        pos = nl + 1;
        return s;
    };
    std::vector<PickleGlobal> globals;
    if (static_cast<unsigned char>(bytes[take(1)]) != 0x80) bad("no PROTO");
    if (static_cast<unsigned char>(bytes[take(1)]) != 2) bad("protocol is not 2");
    while (true) {
        const unsigned char op = static_cast<unsigned char>(bytes[take(1)]);
        switch (op) {
            case ']': case ')': case '(': case 0x88: case 0x89: case 'N': case 't': case 0x85: case 0x86:
            case 0x87: case '}': case 'e': case 'a': case 'l': case 'd': case 'u': case 0x81: case 'b':
            case 'R': case 'Q': case 's':
                break;  // EMPTY_LIST .. SETITEM: no argument
            case 'q': case 'K': case 'h': take(1); break;    // BINPUT, BININT1, BINGET
            case 'M': take(2); break;                        // BININT2
            case 'r': case 'J': case 'j': take(4); break;    // LONG_BINPUT, BININT, LONG_BINGET
            case 'G': take(8); break;                        // BINFLOAT
            case 0x8a:                                       // LONG1: LibTorch reads 8 bytes only
                if (le(1) != 8) bad("LONG1 length is not 8");
                take(8);
                break;
            case 'X': take(le(4)); break;                    // BINUNICODE
            case 0x8d: {                                     // BINUNICODE8
                const std::uint64_t len = le(8);
                if (len > n) bad("BINUNICODE8 length");
                take(static_cast<std::size_t>(len));
            } break;
            case 'c': {                                      // GLOBAL
                PickleGlobal g;
                g.module = line();
                g.name = line();
                globals.push_back(std::move(g));
            } break;
            case '.':                                        // STOP
                return globals;
            default:
                bad("opcode " + std::to_string(op) + " is not one LibTorch's unpickler reads");
        }
    }
}

bool is_custom_class(const std::string& qualname) {
    return qualname.rfind("__torch__.torch.classes.", 0) == 0 || torch::getCustomClass(qualname) != nullptr;
}

void admit_pickle(const std::string& record, const std::string& bytes, const std::set<std::string>& code_classes) {
    const bool debug = ends_with(record, ".debug_pkl");
    for (const PickleGlobal& g : pickle_globals(record, bytes)) {
        const std::string qn = g.module + "." + g.name;
        if (debug) {
            if (qn == "torch.jit._pickle.build_intlist") continue;
            refuse("debug ranges " + record + " name " + qn);
        }
        if (qn == "torch._utils._rebuild_tensor_v2" || qn == "torch.FloatStorage" || qn == "torch.LongStorage" ||
            qn == "collections.OrderedDict") {
            continue;
        }
        if (g.module == "__torch__" || g.module.rfind("__torch__.", 0) == 0) {
            if (is_custom_class(qn)) refuse(record + " restores a custom class " + qn + " (C++ __setstate__)");
            if (code_classes.count(qn) == 0) refuse(record + " restores " + qn + ", which the archive's code does not define");
            continue;
        }
        refuse(record + " uses the global " + qn);
    }
}

// ── step 5: the compiled graphs ──────────────────────────────────────────
std::string node_operator(const torch::jit::Node* n) {
    if (const c10::FunctionSchema* s = n->maybeSchema()) {
        const c10::OperatorName& op = s->operator_name();
        return op.overload_name.empty() ? op.name : op.name + "." + op.overload_name;
    }
    return n->kind().toQualString();
}

// One function's graph as compiled; no pass runs on it. A call is resolved
// when it names, statically, one of the head's own functions:
// prim::CallMethod on a class type (not an interface) whose method is in the
// compilation unit, or prim::CallFunction on a function constant in it. Those
// functions are walked themselves. Every other node is an operator.
struct Walk {
    const std::set<const torch::jit::Function*>& functions;
    std::set<std::string> operators;
    std::set<std::string> unresolved;

    void block(torch::jit::Block* b) {
        for (torch::jit::Node* n : b->nodes()) {
            if (n->kind() == c10::prim::CallMethod || n->kind() == c10::prim::CallFunction) {
                if (!resolved(n)) unresolved.insert(describe(n));
            } else {
                operators.insert(node_operator(n));
            }
            for (torch::jit::Block* sub : n->blocks()) block(sub);
            for (const torch::jit::Symbol& a : n->attributeNames()) {
                if (n->kindOf(a) == torch::jit::AttributeKind::g) block(n->g(a)->block());
                if (n->kindOf(a) == torch::jit::AttributeKind::gs) {
                    for (const std::shared_ptr<torch::jit::Graph>& g : n->gs(a)) block(g->block());
                }
            }
        }
    }

    bool resolved(torch::jit::Node* n) const {
        if (n->inputs().empty()) return false;
        const torch::jit::Function* target = nullptr;
        if (n->kind() == c10::prim::CallMethod) {
            const auto cls = n->input(0)->type()->cast<c10::ClassType>();
            if (!cls) return false;
            target = cls->findMethod(n->s(c10::attr::name));
        } else {
            const torch::jit::Node* constant = n->input(0)->node();
            if (constant->kind() != c10::prim::Constant) return false;
            const auto fn = constant->output()->type()->cast<c10::FunctionType>();
            if (!fn) return false;
            target = fn->function();
        }
        return target != nullptr && functions.count(target) != 0;
    }

    static std::string describe(torch::jit::Node* n) {
        std::string d = n->kind().toQualString();
        if (n->kind() == c10::prim::CallMethod) d += " " + n->s(c10::attr::name);
        if (!n->inputs().empty()) d += " on " + n->input(0)->type()->annotation_str();
        return d;
    }
};

std::string joined(const std::set<std::string>& items) {
    std::string list;
    for (const std::string& item : items) list += (list.empty() ? "" : ", ") + item;
    return list;
}

}  // namespace

void admit_head_archive(const VerifiedModelBytes& bytes, const std::string& libtorch_build) {
    require_reviewed_libtorch(libtorch_build);
    // 1. the loader's own format test, on the same bytes.
    {
        VerifiedModelStream in(bytes);
        if (torch::jit::getFileFormat(in) != torch::jit::FileFormat::ZipFileFormat) {
            refuse("the head is not a ZIP TorchScript archive (FlatBuffer and other formats are refused)");
        }
    }
    try {
        VerifiedModelStream in(bytes);
        caffe2::serialize::PyTorchStreamReader reader(&in);
        // 2. records; 3. code (the class names pickles may restore).
        std::map<std::string, RecordKind> records;
        for (const std::string& name : reader.getAllRecords()) records[name] = record_kind(name);
        if (records.count("data.pkl") == 0 || records.count("constants.pkl") == 0) {
            refuse("the archive has no data.pkl or constants.pkl");
        }
        std::set<std::string> code_classes;
        for (const auto& [name, kind] : records) {
            if (kind != RecordKind::Code) continue;
            for (std::string& c : admit_code(name, record_text(reader, name))) code_classes.insert(std::move(c));
        }
        // 4. pickles the load reads.
        for (const auto& [name, kind] : records) {
            if (kind == RecordKind::Pickle || kind == RecordKind::DebugRanges) {
                admit_pickle(name, record_text(reader, name), code_classes);
            }
        }
    } catch (const DetectorLoadError&) {
        throw;
    } catch (const std::exception& e) {
        refuse(std::string("the archive does not read as TorchScript: ") + std::string(e.what()).substr(0, 200));
    }
}

std::vector<std::string> admit_head_module(const torch::jit::Module& head, const std::string& libtorch_build) {
    require_reviewed_libtorch(libtorch_build);
    const std::set<std::string> allowed(kHeadAdmissionOperators, kHeadAdmissionOperators + kHeadAdmissionOperatorCount);
    std::set<std::string> used;
    try {
        const std::shared_ptr<torch::jit::CompilationUnit> cu = head._ivalue()->compilation_unit();
        if (!cu) refuse("the head has no compilation unit");
        const std::vector<torch::jit::Function*> functions = cu->get_functions();
        if (functions.empty()) refuse("the head compiled no function");
        const std::set<const torch::jit::Function*> own(functions.begin(), functions.end());
        for (torch::jit::Function* fn : functions) {
            const std::string name = fn->qualname().qualifiedName();
            torch::jit::GraphFunction* graph_fn = torch::jit::tryToGraphFunction(*fn);
            if (graph_fn == nullptr) refuse("function " + name + " is not a TorchScript graph");
            graph_fn->ensure_defined();  // compiles only; nothing runs
            Walk walk{own, {}, {}};
            walk.block(graph_fn->graph()->block());
            std::set<std::string> unreviewed;
            for (const std::string& op : walk.operators) {
                if (allowed.count(op) == 0) unreviewed.insert(op);
            }
            if (!unreviewed.empty()) {
                refuse("function " + name + " uses " + joined(unreviewed) + ", not among the reviewed head operators");
            }
            if (!walk.unresolved.empty()) {
                refuse("function " + name + " makes a call that does not resolve to the head's own functions: " +
                       joined(walk.unresolved));
            }
            used.insert(walk.operators.begin(), walk.operators.end());
        }
    } catch (const DetectorLoadError&) {
        throw;
    } catch (const std::exception& e) {
        refuse(std::string("cannot walk the head's functions: ") + std::string(e.what()).substr(0, 200));
    }
    return {used.begin(), used.end()};
}

}  // namespace saccade::shipping
