#pragma once

#include <array>
#include <cstdint>
#include <x86intrin.h>

inline constexpr std::uint64_t fmix64(std::uint64_t k) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdull;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ull;
    k ^= k >> 33;

    return k;
}


std::array<std::size_t, 2> murmur3_64bit(void const* const key, std::size_t const len, std::uint32_t const seed) {

    auto const data = reinterpret_cast<std::uint8_t const*>(key);
    auto const nblocks = len / 16;

    std::uint64_t h1 = seed;
    std::uint64_t h2 = seed;

    constexpr std::uint64_t c1 = 0x87c37b91114253d5ull;
    constexpr std::uint64_t c2 = 0x4cf5ad432745937full;

    //----------
    // body

    auto const blocks = reinterpret_cast<std::uint64_t const*>(data);

    for (std::size_t i = 0; i < nblocks; i++) {
        std::uint64_t k1 = blocks[i*2+0];
        std::uint64_t k2 = blocks[i*2+1];

        k1 *= c1; 
        k1 = _lrotl(k1, 31); 
        k1 *= c2; 
        h1 ^= k1;

        h1 = _lrotl(h1, 27); 
        h1 += h2; 
        h1 = h1 * 5+ 0x52dce729;

        k2 *= c2; 
        k2 = _lrotl(k2, 33); 
        k2 *= c1; 
        h2 ^= k2;

        h2 = _lrotl(h2, 31); 
        h2 += h1; 
        h2 = h2 * 5+ 0x38495ab5;
    }

    //----------
    // tail

    auto const tail = reinterpret_cast<uint8_t const*>(data + nblocks * 16);

    std::uint64_t k1 = 0;
    std::uint64_t k2 = 0;

    switch (len & 15) {
        case 15: k2 ^= ((std::uint64_t)tail[14]) << 48;
        case 14: k2 ^= ((std::uint64_t)tail[13]) << 40;
        case 13: k2 ^= ((std::uint64_t)tail[12]) << 32;
        case 12: k2 ^= ((std::uint64_t)tail[11]) << 24;
        case 11: k2 ^= ((std::uint64_t)tail[10]) << 16;
        case 10: k2 ^= ((std::uint64_t)tail[ 9]) << 8;
        case  9: k2 ^= ((std::uint64_t)tail[ 8]) << 0;
                 k2 *= c2; k2  = _lrotl(k2,33); k2 *= c1; h2 ^= k2;
        case  8: k1 ^= ((std::uint64_t)tail[ 7]) << 56;
        case  7: k1 ^= ((std::uint64_t)tail[ 6]) << 48;
        case  6: k1 ^= ((std::uint64_t)tail[ 5]) << 40;
        case  5: k1 ^= ((std::uint64_t)tail[ 4]) << 32;
        case  4: k1 ^= ((std::uint64_t)tail[ 3]) << 24;
        case  3: k1 ^= ((std::uint64_t)tail[ 2]) << 16;
        case  2: k1 ^= ((std::uint64_t)tail[ 1]) << 8;
        case  1: k1 ^= ((std::uint64_t)tail[ 0]) << 0;
                 k1 *= c1; k1  = _lrotl(k1,31); k1 *= c2; h1 ^= k1;
    };

    //----------
    // finalization

    h1 ^= len; 
    h2 ^= len;

    h1 += h2;
    h2 += h1;

    h1 = fmix64(h1);
    h2 = fmix64(h2);

    h1 += h2;
    h2 += h1;

    return {h1, h1};
}