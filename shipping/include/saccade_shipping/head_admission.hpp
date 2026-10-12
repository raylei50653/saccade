// TorchScript head admission (#549 MB-41, F-1, gap G-1). A head whose archive
// hashes correctly can still read bytes that were never hashed: TorchScript
// can call aten::from_file while it is deserialized (a scripted __setstate__)
// or when it runs (forward, or a torch.jit.fork Subgraph), and
// torch::jit::load also accepts FlatBuffer archives. Gate B (detector_host.hpp)
// therefore admits a head in two steps around torch::jit::load. Any refusal is
// a DetectorLoadError ("head admission (F-1): ...").
//
// Before the load (admit_head_archive), on the verified bytes only. If this
// cannot show that deserialization runs no head code, the head is refused;
// a later check cannot undo a read that already happened.
//   1. The format is ZIP TorchScript, by the loader's own test
//      (torch::jit::getFileFormat). FlatBuffer and unknown formats are refused.
//   2. Every record is one that torch::jit::load(std::istream&) reads: code,
//      its debug ranges, the data and constants pickles and their tensors. Or
//      it is one the loader never reads: version, byteorder, serialization id,
//      or traced inputs (restore_shapes is false). Any other record is refused.
//   3. Every code record parses with the TorchScript parser, the way the
//      importer reads it. It may not hold code the import runs:
//      - no __setstate__: while unpickling, the deserializer calls a class's
//        __setstate__ (import.cpp ObjLoaderFunc);
//      - no parameter default, class constant or other class-body value: the
//        importer evaluates each one by compiling and running it
//        (script_type_parser.cpp evaluateDefaults).
//      Only Module class declarations are admitted. NamedTuple, Enum,
//      interfaces and ordinary classes use other import paths and are refused.
//      A Module body may hold only methods, attribute annotations,
//      __parameters__ / __buffers__ and __annotations__ types.
//   4. Every pickle the load reads uses only opcodes LibTorch's unpickler
//      reads. Its GLOBALs may name only:
//      - in data.pkl and constants.pkl: classes defined in the archive's code
//        (never a registered custom class, whose C++ __setstate__ would run),
//        _rebuild_tensor_v2, the Float/Long storages and OrderedDict;
//      - in debug ranges: build_intlist.
//   Deserialization also compiles the code. Apart from the values step 3
//   refuses, compiling folds only nodes whose inputs and outputs are all
//   immutable (ConstantPropagationImmutableTypes). aten::from_file returns a
//   Tensor, so it never folds, and no head code runs.
//
// After the load (admit_head_module), before anything runs:
//   5. Each function compiled from the archive is walked as compiled; no
//      pass runs on it. That covers every method of every module and class,
//      and every free function. The walk covers nested blocks and
//      graph-valued attributes (prim::fork's Subgraph). Every node must be
//      one of kHeadAdmissionOperators, or a call that resolves statically to
//      another of these functions: prim::CallMethod on a class type, or
//      prim::CallFunction on a function constant. Any other call is refused,
//      for example a method of an interface.
//
// The operator list belongs to the runtime. It is built in here and was
// reviewed against one LibTorch build, kHeadAdmissionLibTorch. No manifest,
// lineage or caller option extends it, and a runtime built against another
// LibTorch refuses every head. The list is exactly the reviewed N02 head's
// operator set (tests/native/test_shipping_head_admission.cpp); none of them
// reads a file.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace torch::jit {
struct Module;
}

namespace saccade::shipping {

class VerifiedModelBytes;

// The LibTorch build the policy was reviewed against.
extern const char* const kHeadAdmissionLibTorch;

// The reviewed operators: "<qualified name>.<overload>" for a node bound to a
// registered operator, else the node kind. Sorted.
extern const char* const kHeadAdmissionOperators[];
extern const std::size_t kHeadAdmissionOperatorCount;

// Steps 1-4. `libtorch_build` is the build this runtime was compiled
// against; any other value refuses every head.
void admit_head_archive(const VerifiedModelBytes& bytes, const std::string& libtorch_build);

// Step 5. Returns the operators the head uses (sorted, distinct), each one of
// kHeadAdmissionOperators.
std::vector<std::string> admit_head_module(const torch::jit::Module& head, const std::string& libtorch_build);

}  // namespace saccade::shipping
