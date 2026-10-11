// A stand-in operator library for the CPU operator-loader controls
// (test_shipping_operator_loader.cpp): built twice, with
// SACCADE_OPERATOR_FIXTURE_VALUE 1 and 2, so two byte-different objects of
// the same size exist. Exported explicitly: the root build hides symbols.
extern "C" __attribute__((visibility("default"))) int saccade_operator_fixture_value() { return SACCADE_OPERATOR_FIXTURE_VALUE; }
