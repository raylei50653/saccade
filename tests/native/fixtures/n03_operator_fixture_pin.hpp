// A fixture N03 pin for CPU tests of the operator loader (#549 S2-2, P-N03
// rev 3d). Linking fixtures/n03_operator_fixture_pin.cpp into a test program
// defines n03_operator_pin() ahead of the production pin library; no runtime
// target links it.
#pragma once

#include <cstdint>
#include <string>

namespace saccade::shipping::testing {

// The pin n03_operator_pin() returns from now on (initially none: every
// request is refused).
void set_fixture_n03_pin(const std::string& sha256, std::int64_t bytes);

}  // namespace saccade::shipping::testing
