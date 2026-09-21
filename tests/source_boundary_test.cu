/* Purpose: Verify default sentence boundaries and exact source profile offsets.
 * Owns: Official vectors, independent expected intervals and invalid input cases.
 * Launch shape: N=1 and N=64 source rows with distinct case positions.
 * Lifetime: One test process without model weights. */
#include "cognitive/source_spans.cuh"
#include "source_boundary_data.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct aotx_boundary_row {
    unsigned char text[AOTX_SOURCE_BYTES];
    unsigned bytes, count, okay, trim, units[AOTX_SOURCE_BYTES];
    aotx_source_span spans[AOTX_SOURCE_SPANS];
};
static unsigned checks, failures;
static void check(bool value, const char *message) {
    ++checks; if (!value) { ++failures; fprintf(stderr, "FAIL: %s\n", message); }
}
static void cuda_check(cudaError_t status) {
    if (status != cudaSuccess) { fprintf(stderr, "%s\n", cudaGetErrorString(status)); exit(2); }
}
__global__ void aotx_boundary_run(aotx_boundary_row *rows, unsigned n) {
    unsigned i = threadIdx.x + blockIdx.x * blockDim.x; if (i >= n) return;
    auto *r = rows + i;
    r->okay = aotx_source_split(r->text, r->bytes, r->units, r->spans, &r->count, r->trim);
}
static void run(aotx_boundary_row *rows, unsigned n) {
    aotx_boundary_run<<<1,64>>>(rows, n); cuda_check(cudaDeviceSynchronize());
}
static void put(aotx_boundary_row &row, const std::string &text, bool trim) {
    memset(&row, 0, sizeof(row)); row.bytes = text.size(); row.trim = trim;
    memcpy(row.text, text.data(), text.size());
}
static void official(aotx_boundary_row *rows, unsigned n) {
    for (unsigned at = 0; at < 512; at += n) {
        for (unsigned i = 0; i < n; ++i) {
            const auto &c = aotx_source_cases[at + i];
            put(rows[i], std::string(c.text, c.bytes), false);
        }
        run(rows, n);
        for (unsigned i = 0; i < n; ++i) {
            const auto &c = aotx_source_cases[at + i]; const auto &r = rows[i];
            check(r.okay && r.count + 1 == c.count, "official boundary count");
            for (unsigned j = 0; j + 1 < c.count && j < r.count; ++j)
                check(r.spans[j].start == c.bounds[j] && r.spans[j].length == c.bounds[j + 1] - c.bounds[j],
                    "official original byte boundary");
        }
    }
}
struct aotx_profile_case { std::string text; std::vector<std::string> spans; };
static void profile(aotx_boundary_row *rows, unsigned n) {
    const std::vector<aotx_profile_case> cases = {
        {"  First.\n\nSecond!\t", {"First.", "Second!"}},
        {"Role here: the motor is ready. It is warm.", {"Role here: the motor is ready.", "It is warm."}},
        {"Dr. Vale is ready.", {"Dr.", "Vale is ready."}},
        {"Value 3.14 is stable. Next.", {"Value 3.14 is stable.", "Next."}},
        {"\"Ready!\" (Next.) Final fragment", {"\"Ready!\"", "(Next.)", "Final fragment"}},
        {"Same. Same.", {"Same.", "Same."}},
        {"Heading\nThe motor is ready", {"Heading", "The motor is ready"}},
        {"\t \n\n", {}}, {"...", {"..."}}, {"", {}},
        {"e\xcc\x81. Next.", {"e\xcc\x81.", "Next."}},
        {"\xe2\x80\x83One.\xe2\x80\x83Two.\xe2\x80\x83", {"One.", "Two."}},
        {"One.\xe2\x80\x8d Next.", {"One.\xe2\x80\x8d", "Next."}},
        {"One.\n\xcc\x81Two.", {"One.", "\xcc\x81Two."}},
        {"A. B. Smith", {"A.", "B.", "Smith"}},
        {"No! really? Yes.", {"No!", "really?", "Yes."}},
        {"  \"quoted\" and \\ paths\tremain.\nTail", {"\"quoted\" and \\ paths\tremain.", "Tail"}},
        {"One.\r\nTwo.\xc2\x85", {"One.", "Two."}}
    };
    for (unsigned at = 0; at < cases.size(); ++at) {
        for (unsigned i = 0; i < n; ++i) put(rows[i], cases[(at + i) % cases.size()].text, true);
        run(rows, n);
        for (unsigned i = 0; i < n; ++i) {
            const auto &c = cases[(at + i) % cases.size()]; const auto &r = rows[i];
            check(r.okay && r.count == c.spans.size(), "profile preserves the expected complete intervals");
            unsigned cursor = 0;
            for (unsigned j = 0; j < c.spans.size() && j < r.count; ++j) {
                auto start = c.text.find(c.spans[j], cursor);
                check(start != std::string::npos && r.spans[j].start == start && r.spans[j].length == c.spans[j].size(),
                    "profile uses the independent original byte offsets");
                cursor = start + c.spans[j].size();
            }
        }
    }
    for (unsigned i = 0; i < n; ++i) {
        std::string text; for (unsigned j = 0; j < 512; ++j) text += "A! ";
        put(rows[i], text, true);
    }
    run(rows, n);
    for (unsigned i = 0; i < n; ++i) check(rows[i].okay && rows[i].count == 512, "exact span count fits");
    for (unsigned i = 0; i < n; ++i) { rows[i].text[rows[i].bytes++] = 'B'; rows[i].text[rows[i].bytes++] = '!'; }
    run(rows, n);
    for (unsigned i = 0; i < n; ++i) check(!rows[i].okay, "one more span refuses without truncation");
    const std::vector<std::string> bad = {"\xc0\x80", "\xed\xa0\x80", "\xf4\x90\x80\x80", "\xe2\x82", "\x80"};
    for (const auto &text : bad) {
        for (unsigned i = 0; i < n; ++i) put(rows[i], text, true);
        run(rows, n);
        for (unsigned i = 0; i < n; ++i) check(!rows[i].okay, "invalid scalar encoding refuses");
    }
}
int main() {
    aotx_boundary_row *rows; cuda_check(cudaMallocManaged(&rows, 64 * sizeof(*rows)));
    for (unsigned n : {1u, 64u}) {
        official(rows, n); profile(rows, n);
        printf("N=%u cumulative checks=%u failures=%u\n", n, checks, failures); fflush(stdout);
    }
    cudaFree(rows); return failures ? 1 : 0;
}
