#include "huffman.h"
#include <algorithm>
#include <limits>
#include <numeric>
#include <queue>
#include <set>
#include <spdlog/spdlog.h>
#include <stack>
#include <tuple>
#include <variant>
#include <vector>
#include <voxcom/utility/binary_reader.h>
#include <voxcom/utility/binary_writer.h>
#include <voxcom/utility/error_handling.h>

using namespace voxcom;

float cost(std::span<const size_t> frequencies, std::span<const uint32_t> codeLengths)
{
    float c = 0.0f;
    for (size_t i = 0; i < frequencies.size(); ++i) {
        c += frequencies[i] * codeLengths[i];
    }
    return c;
}

std::vector<Code> createHuffmanCodeTable(std::span<const size_t> frequencies)
{
    // https://dl.acm.org/doi/pdf/10.1145/3342555
    // See algorithm 1
    struct Node {
        uint32_t left, right;
    };
    constexpr uint32_t Sentinel = std::numeric_limits<uint32_t>::max();
    std::vector<Node> tree;

    using QueueItem = std::pair<size_t, uint32_t>;
    std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<QueueItem>> q;
    for (uint32_t s = 0; s < frequencies.size(); ++s) {
        q.push({ frequencies[s], s });
        tree.push_back({ .left = Sentinel, .right = s });
    }

    while (q.size() > 1) {
        const auto lhs = q.top();
        q.pop();
        const auto rhs = q.top();
        q.pop();

        const auto n = (uint32_t)tree.size();
        tree.push_back({ .left = lhs.second, .right = rhs.second });
        q.push({ lhs.first + rhs.first, n });
    }

    std::vector<std::pair<uint32_t, Code>> codes;
    std::stack<std::pair<Code, uint32_t>> traversalStack;
    traversalStack.push({ Code { .code = 0, .bits = 0 }, q.top().second });
    while (!traversalStack.empty()) {
        const auto item = traversalStack.top();
        const auto& code = item.first;
        const auto& node = tree[item.second];
        traversalStack.pop();

        if (node.left == Sentinel) {
            codes.push_back({ node.right, code });
        } else {
            const Code lhsCode {
                .code = (code.code << 1) | 0b0,
                .bits = code.bits + 1
            };
            traversalStack.push({ lhsCode, node.left });

            const Code rhsCode {
                .code = (code.code << 1) | 0b1,
                .bits = code.bits + 1
            };
            traversalStack.push({ rhsCode, node.right });
        }
    }

    std::sort(std::begin(codes), std::end(codes));
    std::vector<Code> out;
    for (const auto& [_, code] : codes)
        out.push_back(code);

    std::vector<uint32_t> codeLengths;
    for (const auto& code : out)
        codeLengths.push_back(code.bits);
    spdlog::info("cost = {}", cost(frequencies, codeLengths));
    return out;
}

std::vector<Code> assignCodeWords(std::span<const uint32_t> codeLengths)
{
    // https://dl.acm.org/doi/pdf/10.1145/3342555
    // See section 2.5
    std::vector<uint32_t> symbols(codeLengths.size());
    std::iota(std::begin(symbols), std::end(symbols), 0);
    std::sort(std::begin(symbols), std::end(symbols), [&](uint32_t lhs, uint32_t rhs) { return codeLengths[lhs] < codeLengths[rhs]; });

    const uint32_t maxCodeLength = *std::max_element(std::begin(codeLengths), std::end(codeLengths));
    uint32_t L = 0;
    std::vector<Code> codes(symbols.size());
    for (uint32_t symbol : symbols) {
        const uint32_t bits = codeLengths[symbol];
        const uint32_t code = L >> (maxCodeLength - bits);
        codes[symbol] = Code { .code = code, .bits = bits };
        L += 1u << (maxCodeLength - bits);
    }
    return codes;
}

std::vector<uint32_t> createLengthLimitedHuffmanCodeLengths(std::span<const size_t> frequencies, uint32_t maxCodeLength)
{
    // https://dl.acm.org/doi/pdf/10.1145/3342555
    // See algorithm 3
    struct LeafNode {
        uint32_t symbol;
        constexpr auto operator<=>(const LeafNode&) const = default;
    };
    struct Node {
        size_t frequency;
        std::variant<LeafNode, std::vector<Node>> content;

        constexpr bool operator<(const Node& rhs) const
        {
            if (frequency < rhs.frequency)
                return true;
            else if (rhs.frequency < frequency)
                return false;
            return content < rhs.content;
        }
    };
    using Package = std::vector<Node>;

    std::vector<std::set<Node>> packages((size_t)maxCodeLength);
    for (uint32_t symbol = 0; symbol < frequencies.size(); ++symbol)
        packages[0].insert({ frequencies[symbol], LeafNode { .symbol = symbol } });

    for (uint32_t level = 1; level < maxCodeLength; ++level) {
        // Merge adjacent pairs from the previous level.
        const auto& prevLevelPackages = packages[level - 1];
        auto& levelPackages = packages[level];
        for (auto iter = std::begin(prevLevelPackages); iter != std::end(prevLevelPackages);) {
            const Node node1 = *iter++;
            if (iter == std::end(prevLevelPackages))
                break;
            const Node node2 = *iter++;
            levelPackages.insert(Node {
                .frequency = node1.frequency + node2.frequency,
                .content = std::vector<Node> { node1, node2 } });
        }

        // Merge another copy of all leaf nodes into the level
        for (uint32_t symbol = 0; symbol < frequencies.size(); ++symbol)
            levelPackages.insert({ frequencies[symbol], LeafNode { .symbol = symbol } });
    }

    // Copy the smallest 2n-2 items in packages[L-1]
    std::vector<std::vector<Node>> solutions(packages.size());
    const size_t numItems = 2 * frequencies.size() - 2;
    assert_always(numItems <= packages.back().size());
    std::copy_n(std::begin(packages.back()), numItems, std::back_inserter(solutions.back()));

    for (int level = maxCodeLength - 2; level >= 0; --level) {
        // The number of multi-item packages among the items in solution [level+1]
        const size_t count = std::count_if(std::begin(solutions[level + 1]), std::end(solutions[level + 1]), [](const Node& node) { return std::holds_alternative<Package>(node.content); });
        // The smallset 2*count items in packages[level].
        assert_always(2 * count <= packages[level].size());
        std::copy_n(std::begin(packages[level]), 2 * count, std::back_inserter(solutions[level]));
    }

    std::vector<uint32_t> codeLengths(frequencies.size(), 0);
    for (int level = maxCodeLength - 1; level >= 0; --level) {
        for (const auto& node : solutions[level]) {
            if (!std::holds_alternative<LeafNode>(node.content))
                continue;
            const auto symbol = std::get<LeafNode>(node.content).symbol;
            codeLengths[symbol]++;
        }
    }
    return codeLengths;
}

std::vector<Code> createLengthLimitedHuffmanCodeTable(std::span<const size_t> frequencies, uint32_t maxCodeLength)
{
    const auto codeLengths = createLengthLimitedHuffmanCodeLengths(frequencies, maxCodeLength);
    assert_always(kraftSum(codeLengths) <= 1.0f);
    return assignCodeWords(codeLengths);
}

void convertHuffmanToLSB(std::span<Code> codes)
{
    for (Code& code : codes) {
        auto oldCode = code.code;
        code.code = 0;
        for (size_t i = 0; i < code.bits; ++i) {
            const auto oldBit = (oldCode >> i) & 0b1;
            code.code |= oldBit << (code.bits - 1 - i);
        }
    }
}

void HuffmanDecodeLUT::initFromInputCodes()
{
    const uint32_t maxCodeLength = std::max_element(std::begin(m_inputCodes), std::end(m_inputCodes), [](const Code& lhs, const Code& rhs) { return lhs.bits < rhs.bits; })->bits;

    m_bitMask = (1u << maxCodeLength) - 1u;
    m_huffmanCode.resize(1llu << maxCodeLength);
    for (uint32_t key = 0; key < m_inputCodes.size(); ++key) {
        const Code& code = m_inputCodes[key];
        const auto suffixBits = (maxCodeLength - code.bits);
        for (uint32_t suffix = 0; suffix < (1u << suffixBits); ++suffix) {
            const auto combinedCode = (suffix << code.bits) | code.code;
            m_huffmanCode[combinedCode] = { key, code.bits };
        }
    }
}

HuffmanDecodeLUT::HuffmanDecodeLUT(std::span<const Code> codes)
{
    m_inputCodes.resize(codes.size());
    std::copy(std::begin(codes), std::end(codes), std::begin(m_inputCodes));
    initFromInputCodes();
}

void HuffmanDecodeLUT::writeTo(voxcom::BinaryWriter& writer) const
{
    writer.write(m_inputCodes);
}

void HuffmanDecodeLUT::readFrom(voxcom::BinaryReader& reader)
{
    reader.read(m_inputCodes);
    initFromInputCodes();
}

float kraftSum(std::span<const uint32_t> codeLengths)
{
    float sum = 0.0f;
    for (uint32_t codeLength : codeLengths)
        sum += 1.0f / float(1 << codeLength);
    return sum;
}

void testCreateHuffmanCodeTable()
{
    // https://dl.acm.org/doi/pdf/10.1145/3342555
    // See fig 2
    std::vector<size_t> frequencies { 10, 6, 2, 1, 1, 1 };
    const auto codes = createHuffmanCodeTable(frequencies);
    assert_always(codes[0].bits == 1);
    assert_always(codes[1].bits == 2);
    assert_always(codes[2].bits == 4);
    assert_always(codes[3].bits == 4);
    assert_always(codes[4].bits == 4);
    assert_always(codes[5].bits == 4);

    assert_always(codes[0].code == 0b0);
    assert_always(codes[1].code == 0b11);
    assert_always(codes[2].code == 0b1011);
    assert_always(codes[3].code == 0b1000);
    assert_always(codes[4].code == 0b1001);
    assert_always(codes[5].code == 0b1010);
}

void testCreateLengthLimitedHuffmanCodeTable()
{
    // https://dl.acm.org/doi/pdf/10.1145/3342555
    // See fig 6
    std::vector<size_t> frequencies { 20, 17, 6, 3, 2, 2, 2, 1, 1, 1 };
    const auto codeLengths = createLengthLimitedHuffmanCodeLengths(frequencies, 5);
    assert_always(codeLengths[0] == 2);
    assert_always(codeLengths[1] == 2);
    assert_always(codeLengths[2] == 3);
    assert_always(codeLengths[3] == 4);
    assert_always(codeLengths[4] == 4);
    assert_always(codeLengths[5] == 4);
    assert_always(codeLengths[6] == 4);
    assert_always(codeLengths[7] == 5);
    assert_always(codeLengths[8] == 5);
    assert_always(codeLengths[9] == 4);
}

void testAssignCodeWords()
{
    // https://dl.acm.org/doi/pdf/10.1145/3342555
    // See table 1
    std::vector<uint32_t> codeLengths { 1, 2, 4, 5, 5, 5, 5, 5, 6, 6 };
    const auto codes = assignCodeWords(codeLengths);

    for (uint32_t symbol = 0; symbol < codeLengths.size(); ++symbol) {
        assert_always(codes[symbol].bits == codeLengths[symbol]);
    }
    assert_always(codes[0].code == 0b0);
    assert_always(codes[1].code == 0b10);
    assert_always(codes[2].code == 0b1100);
    assert_always(codes[3].code == 0b11010);
    assert_always(codes[4].code == 0b11011);
    assert_always(codes[5].code == 0b11100);
    assert_always(codes[6].code == 0b11101);
    assert_always(codes[7].code == 0b11110);
    assert_always(codes[8].code == 0b111110);
    assert_always(codes[9].code == 0b111111);
}

void testKraftSum()
{
    std::vector<uint32_t> codeLengths { 1, 2, 4, 5, 5, 5, 5, 5, 6, 6 };
    const auto s = kraftSum(codeLengths);
    assert_equal_float(s, 1.0f);
}

void testConvertHuffmanToLSB()
{
    std::vector<uint32_t> codeLengths { 1, 2, 4, 5, 5, 5, 5, 5, 6, 6 };
    auto codes = assignCodeWords(codeLengths);
    convertHuffmanToLSB(codes);
    for (uint32_t symbol = 0; symbol < codeLengths.size(); ++symbol) {
        assert_always(codes[symbol].bits == codeLengths[symbol]);
    }
    assert_always(codes[0].code == 0b0);
    assert_always(codes[1].code == 0b01);
    assert_always(codes[2].code == 0b0011);
    assert_always(codes[3].code == 0b01011);
    assert_always(codes[4].code == 0b11011);
    assert_always(codes[5].code == 0b00111);
    assert_always(codes[6].code == 0b10111);
    assert_always(codes[7].code == 0b01111);
    assert_always(codes[8].code == 0b011111);
    assert_always(codes[9].code == 0b111111);
}

void testHuffmanDecoderLUT()
{
    // https://dl.acm.org/doi/pdf/10.1145/3342555
    // See table 1
    std::vector<uint32_t> codeLengths { 1, 2, 4, 5, 5, 5, 5, 5, 6, 6 };
    auto codes = assignCodeWords(codeLengths);
    convertHuffmanToLSB(codes);
    const HuffmanDecodeLUT decoder { codes };

    {
        const auto [code, bits] = decoder.decode(0b0);
        assert_always(code == 0 && bits == 1);
    }
    {
        const auto [code, bits] = decoder.decode(0b1101010);
        assert_always(code == 0 && bits == 1);
    }

    {
        const auto [code, bits] = decoder.decode(0b01);
        assert_always(code == 1 && bits == 2);
    }
    {
        const auto [code, bits] = decoder.decode(0b0011);
        assert_always(code == 2 && bits == 4);
    }

    {
        const auto [code, bits] = decoder.decode(0b1010011);
        assert_always(code == 2 && bits == 4);
    }
}

void testHuffman()
{
    testKraftSum();
    testAssignCodeWords();
    testCreateHuffmanCodeTable();
    testCreateLengthLimitedHuffmanCodeTable();
    testConvertHuffmanToLSB();
    testHuffmanDecoderLUT();
}
