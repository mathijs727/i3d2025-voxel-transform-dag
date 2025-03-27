#pragma once
#include "voxcom/utility/maths.h"
#include <algorithm>
#include <cassert>
#include <span>
#include <spdlog/spdlog.h>
#include <type_traits>
#include <vector>

#if ENABLE_AVX512
#include <immintrin.h>
#include <intrin.h>
#endif

namespace voxcom {

template <typename Key, typename Value, Key sentinel>
class CuckooHashTable {
public:
    CuckooHashTable() = default;
    CuckooHashTable(std::span<const Key> keys, std::span<const Value> values)
    {
        constexpr double targetLoadFactor = 0.8;
        m_numBuckets = divideRoundUp<size_t>(size_t(keys.size() / targetLoadFactor), keysPerBucket);
        if (m_numBuckets % 2 == 0)
            ++m_numBuckets;
        m_buckets.resize(m_numBuckets);
        for (auto& bucket : m_buckets) {
            for (size_t i = 0; i < keysPerBucket; ++i)
                bucket.keys[i] = sentinel;
        }

        // spdlog::info("{} keys in {} buckets", keys.size(), m_buckets.size());
        // std::vector<Key> uniqueKeys(std::begin(keys), std::end(keys));
        // std::sort(std::begin(uniqueKeys), std::end(uniqueKeys));
        // const auto iter = std::unique(std::begin(uniqueKeys), std::end(uniqueKeys));
        // const size_t numUnique = std::distance(std::begin(uniqueKeys), iter);
        // spdlog::info("Num unique keys: {}", numUnique);
        //
        // std::vector<uint32_t> bucketHistogram(m_buckets.size(), 0);
        // for (const Key key : keys) {
        //     const size_t b1 = hash1(key) % m_numBuckets;
        //     const size_t b2 = hash2(key) % m_numBuckets;
        //     bucketHistogram[b1]++;
        //     bucketHistogram[b2]++;
        // }
        // spdlog::info("Bucket histogram:");
        // for (size_t i = 0; i < m_numBuckets; ++i) {
        //     spdlog::info("[{}] {}", i, bucketHistogram[i]);
        // }

        assert(keys.size() == values.size());
        for (size_t i = 0; i < keys.size(); ++i) {
            insert(keys[i], values[i]);
        }
    }

    void insert(Key key, Value value)
    {
        auto h1 = hash1(key);
        auto h2 = hash2(key);
        auto bucketIdx = h1 % m_numBuckets;
        uint32_t iteration = 0;
        while (true) {
            if (++iteration > 1000)
                spdlog::error("Stuck for key={}, h1={}, h2={}, m_numBuckets={}", key, h1 % m_numBuckets, h2 % m_numBuckets, m_numBuckets);

            // Insert into the first empty slot encountered in the first bucket.
            auto& bucket = m_buckets[bucketIdx];
            if (size_t emptySlotIdx = findKeyInBucket(sentinel, bucket); emptySlotIdx < keysPerBucket) {
                bucket.keys[emptySlotIdx] = key;
                bucket.values[emptySlotIdx] = value;
                return;
            }

            // When the bucket is full: insert by kicking out another item.
            const auto evictedSlotIdx = (h2 + hash2(iteration) * 7727 + 6143) % keysPerBucket;
            std::swap(key, bucket.keys[evictedSlotIdx]);
            std::swap(value, bucket.values[evictedSlotIdx]);

            // Now find a new spot for the item we just kicked out.
            // Regularly switch h1 & h2 so we don't keep inserting into the same bucket.
            h1 = hash1(key);
            h2 = hash2(key);
            const auto oldBucketIdx = bucketIdx;
            bucketIdx = h1 % m_numBuckets;
            if (bucketIdx == oldBucketIdx) {
                std::swap(h1, h2);
                bucketIdx = h1 % m_numBuckets;
            }
        }
    }

    bool find(Key key, Value& value)
    {
#if ENABLE_AVX512
        if constexpr (std::is_same_v<Key, uint64_t>) {
            const __m512i keyAVX512 = _mm512_set1_epi64(key);
            const __m512i sentinelAVX512 = _mm512_set1_epi64(sentinel);

            const auto& bucket0 = m_buckets[hash1(key) % m_numBuckets];
            const __m512i keys0AVX512 = _mm512_load_epi64((const __m512i*)(bucket0.keys));
            const uint32_t hitMask0 = _mm512_cmpeq_epi64_mask(keys0AVX512, keyAVX512);
            if (hitMask0) {
                unsigned long slot = 0;
                _BitScanForward(&slot, hitMask0);
                value = bucket0.values[slot];
                return true;
            }

            if (_mm512_cmpeq_epi64_mask(keys0AVX512, sentinelAVX512))
                return false;

            const auto& bucket1 = m_buckets[hash2(key) % m_numBuckets];
            const __m512i keys1AVX512 = _mm512_load_epi64((const __m512i*)(bucket1.keys));
            const uint32_t hitMask1 = _mm512_cmpeq_epi64_mask(keys1AVX512, keyAVX512);
            if (hitMask1) {
                unsigned long slot = 0;
                _BitScanForward(&slot, hitMask1);
                value = bucket1.values[slot];
                return true;
            }

            return false;
        } else if (std::is_same_v<Key, uint32_t>) {
            const __m512i keyAVX512 = _mm512_set1_epi32(key);
            const __m512i sentinelAVX512 = _mm512_set1_epi32(sentinel);

            const Bucket* buckets[3] = {
                &m_buckets[hash1(key) % m_numBuckets],
                &m_buckets[hash2(key) % m_numBuckets],
                &m_buckets[0]
            };
            const __m512i keys0AVX512 = _mm512_load_epi32((const __m512i*)(buckets[0]->keys));
            const __m512i keys1AVX512 = _mm512_load_epi32((const __m512i*)(buckets[1]->keys));
            const uint32_t hitMask0 = _mm512_cmpeq_epi32_mask(keys0AVX512, keyAVX512);
            const uint32_t hitMask1 = _mm512_cmpeq_epi32_mask(keys1AVX512, keyAVX512);

            const uint64_t combinedHitMask = hitMask0 | (hitMask1 << 16) | (1llu << 32);
            unsigned long combinedSlot = 0;
            _BitScanForward64(&combinedSlot, combinedHitMask);

            const uint32_t slot = combinedSlot & 15; // % 16
            const uint32_t bucket = combinedSlot >> 4; // / 16
            value = buckets[bucket]->values[slot];
            return combinedHitMask & 0xFFFF'FFFF;
        } else {
#endif

            const auto& bucket1 = m_buckets[hash1(key) % m_numBuckets];
            if (auto foundSlotIdx = findKeyInBucket(key, bucket1); foundSlotIdx < keysPerBucket) {
                value = bucket1.values[foundSlotIdx];
                return true;
            }

            const auto& bucket2 = m_buckets[hash2(key) % m_numBuckets];
            if (auto foundSlotIdx = findKeyInBucket(key, bucket2); foundSlotIdx < keysPerBucket) {
                value = bucket2.values[foundSlotIdx];
                return true;
            }

            return false;
#if ENABLE_AVX512
        }
#endif
    }

private:
    static constexpr size_t keyBytesPerBucket = 64;
    static constexpr size_t keysPerBucket = keyBytesPerBucket / sizeof(Key);
    static_assert(keyBytesPerBucket % sizeof(Key) == 0);
    static_assert(std::is_integral_v<Key>);

    struct alignas(64) Bucket {
        Key keys[keysPerBucket];
        Value values[keysPerBucket];
    };
    std::vector<Bucket> m_buckets;
    size_t m_numBuckets;

private:
    static inline uint32_t findKeyInBucket(Key key, const Bucket& bucket)
    {
        for (uint32_t i = 0; i < keysPerBucket; ++i) {
            if (bucket.keys[i] == key)
                return i;
        }
        return keysPerBucket;
    }

    static inline size_t hash1(Key key)
    {
        // https://gist.github.com/badboy/6267743
        return (key | 64) ^ ((key >> 15) | (key << 17));
    }
    static inline size_t hash2(Key key)
    {
        return 1103515245 * key + 12345;
    }
};
}