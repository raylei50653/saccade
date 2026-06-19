#include "tracking/auction.hpp"
#include "tracking/hungarian.hpp"
#include "tracking/sinkhorn.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

void fail(const std::string& message) {
    std::cerr << message << std::endl;
    std::exit(1);
}

void expect_true(bool condition, const std::string& message) {
    if (!condition) {
        fail(message);
    }
}

void expect_assignment(
    const std::vector<int>& actual,
    const std::vector<int>& expected,
    const std::string& label
) {
    if (actual != expected) {
        std::ostringstream oss;
        oss << label << " expected [";
        for (size_t i = 0; i < expected.size(); ++i) {
            if (i > 0) {
                oss << ", ";
            }
            oss << expected[i];
        }
        oss << "] but got [";
        for (size_t i = 0; i < actual.size(); ++i) {
            if (i > 0) {
                oss << ", ";
            }
            oss << actual[i];
        }
        oss << "]";
        fail(oss.str());
    }
}

void test_hungarian_square_matrix() {
    saccade::HungarianAlgorithm algorithm;
    std::vector<std::vector<float>> cost_matrix{
        {4.0f, 1.0f, 3.0f},
        {2.0f, 0.0f, 5.0f},
        {3.0f, 2.0f, 2.0f},
    };
    std::vector<int> assignment;
    algorithm.Solve(cost_matrix, assignment);
    expect_assignment(assignment, {1, 0, 2}, "hungarian square matrix");
}

void test_hungarian_transposed_case() {
    saccade::HungarianAlgorithm algorithm;
    std::vector<std::vector<float>> cost_matrix{
        {10.0f, 1.0f},
        {1.0f, 10.0f},
        {2.0f, 3.0f},
    };
    std::vector<int> assignment;
    algorithm.Solve(cost_matrix, assignment);

    expect_true(assignment.size() == 3, "hungarian transposed size mismatch");
    expect_true(assignment[0] == 1, "hungarian transposed expected row 0 -> col 1");
    expect_true(assignment[1] == 0, "hungarian transposed expected row 1 -> col 0");
    expect_true(assignment[2] == -1, "hungarian transposed expected unmatched row");
}

void test_hungarian_empty_columns() {
    saccade::HungarianAlgorithm algorithm;
    std::vector<std::vector<float>> cost_matrix{
        {},
        {},
    };
    std::vector<int> assignment{42};
    algorithm.Solve(cost_matrix, assignment);
    expect_assignment(assignment, {-1, -1}, "hungarian empty columns");
}

void test_auction_square_matrix() {
    std::vector<std::vector<float>> profit_matrix{
        {10.0f, 1.0f, 1.0f},
        {1.0f, 10.0f, 1.0f},
        {1.0f, 1.0f, 10.0f},
    };
    std::vector<int> assignment;
    saccade::AuctionAlgorithm::Solve(profit_matrix, assignment, 0.01f);
    expect_assignment(assignment, {0, 1, 2}, "auction square matrix");
}

void test_auction_rectangular_matrix() {
    std::vector<std::vector<float>> profit_matrix{
        {9.0f, 1.0f},
        {1.0f, 8.0f},
        {2.0f, 3.0f},
    };
    std::vector<int> assignment;
    saccade::AuctionAlgorithm::Solve(profit_matrix, assignment, 0.01f);

    expect_true(assignment.size() == 3, "auction rectangular size mismatch");
    expect_true(assignment[0] == 0, "auction rectangular expected row 0 -> col 0");
    expect_true(assignment[1] == 1, "auction rectangular expected row 1 -> col 1");
    expect_true(assignment[2] == -1, "auction rectangular expected unmatched row");
}

void test_auction_contested_loser_rebids() {
    std::vector<std::vector<float>> profit_matrix{
        {10.0f, 9.0f},
        {10.0f, 1.0f},
    };
    std::vector<int> assignment;
    saccade::AuctionAlgorithm::Solve(profit_matrix, assignment, 0.01f);

    expect_assignment(assignment, {1, 0}, "auction contested loser rebids");
}

void test_auction_empty_columns() {
    std::vector<std::vector<float>> profit_matrix{
        {},
        {},
    };
    std::vector<int> assignment{9};
    saccade::AuctionAlgorithm::Solve(profit_matrix, assignment, 0.01f);
    expect_assignment(assignment, {-1, -1}, "auction empty columns");
}

void test_sinkhorn_square_matrix() {
    std::vector<std::vector<float>> cost_matrix{
        {0.0f, 5.0f, 5.0f},
        {5.0f, 0.0f, 5.0f},
        {5.0f, 5.0f, 0.0f},
    };
    std::vector<int> assignment;
    saccade::SinkhornAlgorithm::Solve(cost_matrix, assignment, 10.0f, 100);
    expect_assignment(assignment, {0, 1, 2}, "sinkhorn square matrix");
}

void test_sinkhorn_empty_columns() {
    std::vector<std::vector<float>> cost_matrix{
        {},
        {},
    };
    std::vector<int> assignment{7};
    saccade::SinkhornAlgorithm::Solve(cost_matrix, assignment, 10.0f, 5);
    expect_assignment(assignment, {-1, -1}, "sinkhorn empty columns");
}

void test_sinkhorn_rectangular_matrix() {
    std::vector<std::vector<float>> cost_matrix{
        {0.0f, 6.0f},
        {6.0f, 0.0f},
        {3.0f, 4.0f},
    };
    std::vector<int> assignment;
    saccade::SinkhornAlgorithm::Solve(cost_matrix, assignment, 12.0f, 120);

    expect_true(assignment.size() == 3, "sinkhorn rectangular size mismatch");
    expect_true(assignment[0] == 0, "sinkhorn rectangular expected row 0 -> col 0");
    expect_true(assignment[1] == 1, "sinkhorn rectangular expected row 1 -> col 1");
    expect_true(assignment[2] == -1, "sinkhorn rectangular expected unmatched row");
}

}  // namespace

int main() {
    test_hungarian_square_matrix();
    test_hungarian_transposed_case();
    test_hungarian_empty_columns();

    test_auction_square_matrix();
    test_auction_rectangular_matrix();
    test_auction_contested_loser_rebids();
    test_auction_empty_columns();

    test_sinkhorn_square_matrix();
    test_sinkhorn_empty_columns();
    test_sinkhorn_rectangular_matrix();

    std::cout << "native assignment algorithm tests passed" << std::endl;
    return 0;
}
