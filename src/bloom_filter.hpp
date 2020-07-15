#pragma once

#include "murmur3.hpp"

#include <bitset>
#include <cmath>
#include <functional>
#include <limits>
#include <ratio>
#include <tuple>
#include <type_traits>

namespace detail {

//
template<class T, class U, class R = std::common_type_t<T, std::decay_t<decltype(std::declval<T const&>() % std::declval<U const&>())>>>
inline constexpr R fast_mod(T const& input, U const& ceil) noexcept {
    return input >= ceil ? input % ceil : input;
}

//
template<class Hash>
struct is_transparent {
    template<class U>
    static auto test(int) -> decltype(U::is_transparent, std::true_type{});

    template<class>
    static std::false_type test(...);

    static constexpr bool value = decltype(test<Hash>(0))::value;
};

template<class Hash>
inline static constexpr bool is_transparent_v = is_transparent<Hash>::value;

//
template<class T>
struct is_tuple_like {
    template<class U>
    static auto test_tuple_size(int) -> decltype(std::tuple_size_v<U>, std::true_type{});
    template<class>
    static std::false_type test_tuple_size(...);

    template<class U>
    static auto test_get(int) -> decltype(std::get<0>(std::declval<U>()), std::true_type{});
    template<class>
    static std::false_type test_get(...);
    
    static constexpr bool value = decltype(test_tuple_size<T>(0))::value && decltype(test_get<T>(0))::value;
};

template<class T>
inline static constexpr bool is_tuple_like_v = is_tuple_like<T>::value;

//
template<class H, class Arg>
struct is_valid_hash {
    static constexpr bool value = std::conjunction_v<std::is_default_constructible<H>, std::is_invocable<H, Arg const&>, 
        is_tuple_like<std::invoke_result_t<H, Arg const&>>>;
};

template<class H, class Arg>
inline static constexpr bool is_valid_hash_v = is_valid_hash<H, Arg>::value;

//
template<class T, std::size_t Expected, class FalsePositiveRatio, class Hash>
struct bloom_filter_false_positive_policy;

template<class T, std::size_t Expected, std::intmax_t Num, std::intmax_t Denom, class Hash>
struct bloom_filter_false_positive_policy<T, Expected, std::ratio<Num, Denom>, Hash> {

    using value_type = T;
    using hash_type = Hash;

    static constexpr std::size_t expected() noexcept {
        return Expected;
    }

    static constexpr double false_positive_probability() noexcept { 
        return static_cast<double>(Num) / static_cast<double>(Denom); 
    }

    static constexpr std::size_t num_buckets() noexcept {
        return static_cast<std::size_t>(-((static_cast<double>(Expected) * std::log(false_positive_probability())) / (std::pow(std::log(2), 2))));
    };

    static constexpr std::size_t num_hashes() noexcept {
        using Tpl = std::invoke_result_t<Hash, T const&>;
        return std::tuple_size_v<Tpl>;
    }

    static_assert(is_valid_hash_v<Hash, T>, "Invalid hash type");
    static_assert(false_positive_probability() > 0. && false_positive_probability() < 1., 
        "bloom filter false_positive_probability() must be between [0, 1]");
};

//
template<class T, std::size_t Expected, std::size_t NumBuckets, class Hash>
struct bloom_filter_size_policy {

    using value_type = T;
    using hash_type = Hash;

    static constexpr std::size_t expected() noexcept {
        return Expected;
    }

    static constexpr double false_positive_probability() noexcept {
        double x = 1. - (1. / static_cast<double>(NumBuckets));
        x = 1. - std::pow(x, (static_cast<double>(num_hashes()) * static_cast<double>(Expected)));
        return std::pow(x, static_cast<double>(num_hashes()));
    }

    static constexpr std::size_t num_buckets() noexcept {
        return NumBuckets;
    }

    static constexpr std::size_t num_hashes() noexcept {
        using Tpl = std::invoke_result_t<Hash, T const&>;
        return std::tuple_size_v<Tpl>;
    }

    static_assert(is_valid_hash_v<Hash, T>, "Invalid hash type");
    static_assert(false_positive_probability() > 0. && false_positive_probability() < 1., 
        "bloom filter false_positive_probability() must be between [0, 1]");
};

//
inline static constexpr std::size_t unknown_expected = static_cast<std::size_t>(-1);
inline static constexpr double unknown_false_positive_probability = std::numeric_limits<double>::infinity();

template<class T, std::size_t NumBuckets, class Hash>
struct bloom_filter_minimal_policy {

    using value_type = T;
    using hash_type = Hash;

    static constexpr std::size_t expected() noexcept {
        return unknown_expected;
    }

    static constexpr double false_positive_probability() noexcept {
        return unknown_false_positive_probability;
    }

    static constexpr std::size_t num_buckets() noexcept {
        return NumBuckets;
    }

    static constexpr std::size_t num_hashes() noexcept {
        using Tpl = std::invoke_result_t<Hash, T const&>;
        return std::tuple_size_v<Tpl>;
    }

    static_assert(is_valid_hash_v<Hash, T>, "Invalid hash type");
};

} // namespace detail

template<class T, std::enable_if_t<std::is_scalar_v<T>, int> = 0>
struct default_bloom_filter_hash {
    constexpr auto operator()(T const& t) const noexcept {
        return murmur3_64bit(reinterpret_cast<T const*>(std::addressof(t)), sizeof(T), 5342357u);
    }
};

template<class Policy, std::size_t BitsPerBucket>
class counting_bloom_filter {
public:

    using value_type = typename Policy::value_type;
    using hash_type = typename Policy::hash_type;
    
    static constexpr auto unknown_expected = detail::unknown_expected;
    static constexpr auto unknown_false_positive_probability = detail::unknown_false_positive_probability;

    constexpr counting_bloom_filter() noexcept = default;

    constexpr void insert(value_type const& val) noexcept {
        auto const indices = hash_type{}(val);
        std::apply([this](auto&&... idx){
            auto increment_bucket = [this](auto&& i) {
                auto& bucket = buckets_[detail::fast_mod(i, num_buckets())];
                bucket = bucket.to_ullong() + 1;
            };
            (increment_bucket(idx), ...);
        }, indices);
    }

    constexpr void remove(value_type const& val) noexcept {
        auto const indices = hash_type{}(val);
        std::apply([this](auto&&... idx){
            auto decrement_bucket = [this](auto&& i) {
                auto& bucket = buckets_[detail::fast_mod(i, num_buckets())];
                bucket = bucket.to_ullong() - 1;
            };
            (decrement_bucket(idx), ...);
        }, indices);
    }

    [[nodiscard]] constexpr bool contains(value_type const& val) const noexcept {
        auto const indices = hash_type{}(val);
        return std::apply([this](auto&&... idx) -> bool {
            return ((buckets_[detail::fast_mod(idx, num_buckets())].to_ullong() > 0ull) && ...);
        }, indices);
    }

    template<class U, class Hash = hash_type, std::enable_if_t<detail::is_transparent_v<Hash>, int> = 0>
    [[nodiscard]] constexpr bool contains(U const& val) const noexcept {
        auto const indices = hash_type{}(val);
        return std::apply([this](auto&&... idx) -> bool {
            return ((buckets_[detail::fast_mod(idx, num_buckets())].to_ullong() > 0ull) && ...);
        }, indices);
    }

    [[nodiscard]] constexpr bool contains_less_than(value_type const& val, std::size_t const count) const noexcept {
        auto const indices = hash_type{}(val);
        return std::apply([&](auto&&... idx) -> bool {
            return ((buckets_[detail::fast_mod(idx, num_buckets())].to_ullong() < count) && ...);
        }, indices);
    }

    template<class U, class Hash = hash_type, std::enable_if_t<detail::is_transparent_v<Hash>, int> = 0>
    [[nodiscard]] constexpr bool contains_less_than(U const& val, std::size_t const count) const noexcept {
        auto const indices = hash_type{}(val);
        return std::apply([&](auto&&... idx) -> bool {
            return ((buckets_[detail::fast_mod(idx, num_buckets())].to_ullong() < count) && ...);
        }, indices);
    }

    constexpr void reset() noexcept {
        buckets_.fill(0);
    }

    [[nodiscard]] constexpr std::size_t approx_inserted() const noexcept {
        std::size_t buckets_set = 0;
        for (auto&& bucket : buckets_)
            if (bucket.to_ullong() > 0ull)
                ++buckets_set;

        return static_cast<std::size_t>(
            -((static_cast<double>(num_buckets()) / static_cast<double>(num_hashes())) * 
            (std::log(1. - (static_cast<double>(buckets_set) / static_cast<double>(num_buckets())))))
        );
    }

    [[nodiscard]] static constexpr std::size_t expected() noexcept {
        return Policy::expected();
    }

    [[nodiscard]] static constexpr double false_positive_probability() noexcept {
        return Policy::false_positive_probability();
    }

    [[nodiscard]] static constexpr std::size_t num_buckets() noexcept {
        return Policy::num_buckets();
    }

    [[nodiscard]] static constexpr std::size_t num_hashes() noexcept {
        return Policy::num_hashes();
    }

    [[nodiscard]] static constexpr std::size_t max_bucket_value() noexcept {
        return std::bitset<BitsPerBucket>().set().to_ullong();
    }

private:
    std::array<std::bitset<BitsPerBucket>, num_buckets()> buckets_;
};


template<class Policy>
class counting_bloom_filter<Policy, 0> {
public:

    using value_type = typename Policy::value_type;
    using hash_type = typename Policy::hash_type;
    
    static constexpr auto unknown_expected = detail::unknown_expected;
    static constexpr auto unknown_false_positive_probability = detail::unknown_false_positive_probability;

    constexpr counting_bloom_filter() noexcept = default;

    constexpr void insert(value_type const& t) noexcept {
        auto const pos = hash_type{}(t);
        std::apply([this](auto&&... idx){
            (bits_.set(detail::fast_mod(idx, num_buckets())), ...);
        }, pos);
    }

    [[nodiscard]] constexpr bool contains(value_type const& t) const noexcept {
        auto const pos = hash_type{}(t);
        return std::apply([this](auto&&... idx) -> bool {
            return (bits_.test(detail::fast_mod(idx, num_buckets())) && ...);
        }, pos);
    }

    template<class K, class Hash = hash_type, std::enable_if_t<detail::is_transparent_v<Hash>, int> = 0>
    [[nodiscard]] constexpr bool contains(K const& k) const noexcept {
        auto const pos = hash_type{}(k);
        return std::apply([this](auto&&... idx) -> bool {
            return (bits_.test(detail::fast_mod(idx, num_buckets())) && ...);
        }, pos);
    }

    constexpr void reset() noexcept {
        bits_.reset();
    }

    [[nodiscard]] constexpr std::size_t approx_inserted() const noexcept {
        auto const bits_set = static_cast<double>(bits_.count());
        return static_cast<std::size_t>(
            -((static_cast<double>(num_buckets()) / static_cast<double>(num_hashes())) * 
            (std::log(1. - (bits_set / static_cast<double>(num_buckets())))))
        );
    }

    [[nodiscard]] static constexpr std::size_t expected() noexcept {
        return Policy::expected();
    }

    [[nodiscard]] static constexpr double false_positive_probability() noexcept {
        return Policy::false_positive_probability();
    }

    [[nodiscard]] static constexpr std::size_t num_buckets() noexcept {
        return Policy::num_buckets();
    }

    [[nodiscard]] static constexpr std::size_t num_hashes() noexcept {
        return Policy::num_hashes();
    }

    [[nodiscard]] static constexpr std::size_t max_bucket_value() noexcept {
        return 1;
    }

private:
    std::bitset<num_buckets()> bits_;
};

template<std::size_t N>
struct expected_t { static constexpr auto value = N; };

template<std::size_t N>
inline static constexpr expected_t<N> expected {};

template<std::size_t N>
struct num_buckets_t { static constexpr auto value = N; };

template<std::size_t N>
inline static constexpr num_buckets_t<N> num_buckets {};

template<std::size_t N>
struct num_bits_t { static constexpr auto value = N; };

template<std::size_t N>
inline static constexpr num_bits_t<N> num_bits{};

template<std::size_t N>
struct bits_per_bucket_t { static constexpr auto value = N; };

template<std::size_t N>
inline static constexpr bits_per_bucket_t<N> bits_per_bucket {};

template<class T, class Hash, std::size_t Expected, std::intmax_t Num, std::intmax_t Denom> 
constexpr auto make_bloom_filter(expected_t<Expected> const, std::ratio<Num, Denom> const) noexcept {
    return counting_bloom_filter<detail::bloom_filter_false_positive_policy<T, Expected, std::ratio<Num, Denom>, Hash>, 0>();
}

template<class T, class Hash, std::size_t Expected, std::size_t NumBuckets>
constexpr auto make_bloom_filter(expected_t<Expected> const, num_bits_t<NumBuckets> const) noexcept {
    return counting_bloom_filter<detail::bloom_filter_size_policy<T, Expected, NumBuckets, Hash>, 0>();
}

template<class T, class Hash, std::size_t NumBuckets>
constexpr auto make_bloom_filter(num_bits_t<NumBuckets> const) noexcept {
    return counting_bloom_filter<detail::bloom_filter_minimal_policy<T, NumBuckets, Hash>, 0>();
}

template<class T, class Hash, std::size_t BitsPerBucket, std::size_t Expected, std::intmax_t Num, std::intmax_t Denom>
constexpr auto make_counting_bloom_filter(bits_per_bucket_t<BitsPerBucket> const, expected_t<Expected> const, std::ratio<Num, Denom> const) noexcept {
    return counting_bloom_filter<detail::bloom_filter_false_positive_policy<T, Expected, std::ratio<Num, Denom>, Hash>, BitsPerBucket>();                                           
}

template<class T, class Hash, std::size_t BitsPerBucket, std::size_t Expected, std::size_t NumBuckets>
constexpr auto make_counting_bloom_filter(bits_per_bucket_t<BitsPerBucket> const, expected_t<Expected> const, num_buckets_t<NumBuckets> const) noexcept {
    return counting_bloom_filter<detail::bloom_filter_size_policy<T, Expected, NumBuckets, Hash>, BitsPerBucket>();
}

template<class T, class Hash, std::size_t BitsPerBucket, std::size_t NumBuckets>
constexpr auto make_counting_bloom_filter(bits_per_bucket_t<BitsPerBucket> const, num_buckets_t<NumBuckets> const) noexcept {
    return counting_bloom_filter<detail::bloom_filter_minimal_policy<T, NumBuckets, Hash>, BitsPerBucket>();
}