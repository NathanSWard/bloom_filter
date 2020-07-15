#include "../src/bloom_filter.hpp"
#include <cassert>

int main() {
    auto bf = make_bloom_filter<int, default_bloom_filter_hash<int>>(expected<1000>, std::milli{});
    bf.insert(42);
    bf.insert(0);

    assert(bf.contains(42));
    assert(bf.contains(0));

    assert(!bf.contains(100));

}