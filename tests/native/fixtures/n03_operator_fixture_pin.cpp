// See n03_operator_fixture_pin.hpp.
#include "n03_operator_fixture_pin.hpp"

#include "saccade_shipping/operator_loader.hpp"

namespace saccade::shipping {
namespace {

std::string fixture_sha256;
N03OperatorPin fixture_pin{"", -1};

}  // namespace

const N03OperatorPin& n03_operator_pin() { return fixture_pin; }

void testing::set_fixture_n03_pin(const std::string& sha256, std::int64_t bytes) {
    fixture_sha256 = sha256;
    fixture_pin = {fixture_sha256.c_str(), bytes};
}

}  // namespace saccade::shipping
