// A stand-in operator library for the CPU operator-loader and admission
// controls (test_shipping_operator_loader.cpp, test_shipping_n03_admission.cpp):
// built with SACCADE_OPERATOR_FIXTURE_VALUE 1 and 2, so two byte-different
// objects of the same size exist. Exported explicitly: the root build hides
// symbols. Its constructor sets SACCADE_OPERATOR_FIXTURE_CONSTRUCTED, the
// sentinel that a refusal came before any operator constructor ran.
#include <stdlib.h>

extern "C" __attribute__((visibility("default"))) int saccade_operator_fixture_value() { return SACCADE_OPERATOR_FIXTURE_VALUE; }

__attribute__((constructor)) static void saccade_operator_fixture_constructed() {
    ::setenv("SACCADE_OPERATOR_FIXTURE_CONSTRUCTED", "1", 1);
}
