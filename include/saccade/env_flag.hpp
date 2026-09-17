#pragma once

#include <cstdlib>
#include <cstring>

namespace saccade {

/// Fail-closed diagnostic env flag.
/// Unset/empty, 0/false/no/off → false. Only 1/true/yes/on → true.
/// Unknown tokens stay off. Keep tokens in sync with
/// ``saccade.perception.eval.assoc_stats_env``.
inline bool env_diagnostic_on(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') return false;
    auto eq = [value](const char* lit) {
        return std::strcmp(value, lit) == 0;
    };
    if (eq("0") || eq("false") || eq("False") || eq("FALSE")
        || eq("no") || eq("No") || eq("NO")
        || eq("off") || eq("Off") || eq("OFF")) {
        return false;
    }
    return eq("1") || eq("true") || eq("True") || eq("TRUE")
        || eq("yes") || eq("Yes") || eq("YES")
        || eq("on") || eq("On") || eq("ON");
}

}  // namespace saccade
