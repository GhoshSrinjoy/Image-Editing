/*
 * Parallel PNG encoder.
 *
 * The image is cut into bands of rows. Every band is filtered and deflated on
 * its own thread into a run of deflate blocks that ends on a byte boundary (a
 * sync flush), so the bands concatenate into a single zlib stream, the way
 * pigz does it. Each band is written as its own IDAT chunk, which lets the
 * workers compute their chunk CRCs too; the per-band Adler-32 sums are merged
 * at the end.
 */
#include "imgedit.h"
#include "parallel.h"

#include <stdlib.h>
#include <string.h>

#if defined(_MSC_VER)
#include <intrin.h>
static int ctz64(uint64_t v)
{
    unsigned long i;
    _BitScanForward64(&i, v);
    return (int)i;
}
#else
#define ctz64(v) __builtin_ctzll(v)
#endif

/* ---- checksums --------------------------------------------------------- */

static uint32_t crc32_update(const uint32_t (*crc_tab)[256], uint32_t crc, const uint8_t *p, size_t n)
{
    crc = ~crc;
    while (n >= 8) {
        uint32_t a = crc ^ ((uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24);
        uint32_t b = (uint32_t)p[4] | (uint32_t)p[5] << 8 | (uint32_t)p[6] << 16 | (uint32_t)p[7] << 24;
        crc = crc_tab[7][a & 255] ^ crc_tab[6][(a >> 8) & 255] ^ crc_tab[5][(a >> 16) & 255] ^
              crc_tab[4][a >> 24] ^ crc_tab[3][b & 255] ^ crc_tab[2][(b >> 8) & 255] ^
              crc_tab[1][(b >> 16) & 255] ^ crc_tab[0][b >> 24];
        p += 8;
        n -= 8;
    }
    while (n--) crc = crc_tab[0][(crc ^ *p++) & 255] ^ (crc >> 8);
    return ~crc;
}

static uint32_t adler32(const uint8_t *p, size_t n)
{
    uint32_t a = 1, b = 0;
    while (n) {
        size_t k = n < 5552 ? n : 5552;
        n -= k;
        while (k >= 8) {
            a += p[0]; b += a; a += p[1]; b += a; a += p[2]; b += a; a += p[3]; b += a;
            a += p[4]; b += a; a += p[5]; b += a; a += p[6]; b += a; a += p[7]; b += a;
            p += 8;
            k -= 8;
        }
        while (k--) {
            a += *p++;
            b += a;
        }
        a %= 65521;
        b %= 65521;
    }
    return b << 16 | a;
}

/* Adler-32 of A||B from adler(A), adler(B) and len(B) (same formula as zlib). */
static uint32_t adler32_combine(uint32_t a1, uint32_t a2, size_t len2)
{
    const uint64_t BASE = 65521;
    uint64_t rem = len2 % BASE;
    uint64_t s1 = a1 & 0xffff;
    uint64_t s2 = (rem * s1) % BASE;
    s1 += (a2 & 0xffff) + BASE - 1;
    s2 += ((a1 >> 16) & 0xffff) + ((a2 >> 16) & 0xffff) + BASE - rem;
    if (s1 >= BASE) s1 -= BASE;
    if (s1 >= BASE) s1 -= BASE;
    if (s2 >= BASE << 1) s2 -= BASE << 1;
    if (s2 >= BASE) s2 -= BASE;
    return (uint32_t)(s1 | s2 << 16);
}

/* ---- deflate tables ---------------------------------------------------- */

static const uint16_t len_base[29] = {3,  4,  5,  6,  7,  8,  9,  10, 11,  13,  15,  17,  19,  23, 27,
                                      31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
static const uint8_t len_extra[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2,
                                      2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
static const uint16_t dist_base[30] = {1,   2,   3,   4,   5,   7,    9,    13,   17,   25,
                                       33,  49,  65,  97,  129, 193,  257,  385,  513,  769,
                                       1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
static const uint8_t dist_extra[30] = {0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6,
                                       6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};
static const uint8_t cl_order[19] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};

/* Built once per encode so concurrent encodes never share mutable state. */
typedef struct {
    uint32_t crc[8][256];
    uint8_t len_code[259];   /* match length -> length code 0..28 */
    uint8_t dist_small[256]; /* distance-1 (< 256) -> distance code */
    uint8_t dist_large[256]; /* (distance-1) >> 7 -> distance code */
    uint8_t fixed_ll_len[288], fixed_d_len[30];
    uint16_t fixed_ll_code[288], fixed_d_code[30];
} ztables;

static void huff_codes(const uint8_t *lens, int n, uint16_t *codes);

static void tables_init(ztables *t)
{
    uint32_t (*crc_tab)[256] = t->crc;
    uint8_t *len_code = t->len_code, *dist_small = t->dist_small, *dist_large = t->dist_large;
    uint8_t *fixed_ll_len = t->fixed_ll_len, *fixed_d_len = t->fixed_d_len;
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
        crc_tab[0][i] = c;
    }
    for (int i = 0; i < 256; i++)
        for (int k = 1; k < 8; k++) crc_tab[k][i] = (crc_tab[k - 1][i] >> 8) ^ crc_tab[0][crc_tab[k - 1][i] & 255];

    for (int c = 0; c < 29; c++) {
        int span = c == 28 ? 1 : 1 << len_extra[c];
        for (int l = len_base[c]; l < len_base[c] + span && l <= 258; l++) len_code[l] = (uint8_t)c;
    }
    for (int c = 0; c < 30; c++)
        for (int d = dist_base[c]; d < dist_base[c] + (1 << dist_extra[c]); d++) {
            if (d - 1 < 256) dist_small[d - 1] = (uint8_t)c;
            else dist_large[(d - 1) >> 7] = (uint8_t)c;
        }
    for (int i = 0; i < 288; i++) fixed_ll_len[i] = i < 144 ? 8 : i < 256 ? 9 : i < 280 ? 7 : 8;
    for (int i = 0; i < 30; i++) fixed_d_len[i] = 5;
    huff_codes(fixed_ll_len, 288, t->fixed_ll_code);
    huff_codes(fixed_d_len, 30, t->fixed_d_code);
}

static inline int dist_code(const ztables *t, int d)
{
    return d <= 256 ? t->dist_small[d - 1] : t->dist_large[(d - 1) >> 7];
}

/* ---- Huffman code construction ----------------------------------------- */

typedef struct {
    uint32_t key;
    uint16_t sym;
} hsym;

static int hsym_cmp(const void *a, const void *b)
{
    uint32_t x = ((const hsym *)a)->key, y = ((const hsym *)b)->key;
    return x < y ? -1 : x > y;
}

/* In-place minimum-redundancy code lengths (Moffat & Katajainen) over symbols
 * sorted by ascending weight; on return A[i].key holds the length. */
static void min_redundancy(hsym *A, int n)
{
    int root, leaf, next, avbl, used, dpth;
    if (n == 1) {
        A[0].key = 1;
        return;
    }
    A[0].key += A[1].key;
    root = 0;
    leaf = 2;
    for (next = 1; next < n - 1; next++) {
        if (leaf >= n || A[root].key < A[leaf].key) {
            A[next].key = A[root].key;
            A[root++].key = (uint32_t)next;
        } else {
            A[next].key = A[leaf++].key;
        }
        if (leaf >= n || (root < next && A[root].key < A[leaf].key)) {
            A[next].key += A[root].key;
            A[root++].key = (uint32_t)next;
        } else {
            A[next].key += A[leaf++].key;
        }
    }
    A[n - 2].key = 0;
    for (next = n - 3; next >= 0; next--) A[next].key = A[A[next].key].key + 1;
    avbl = 1;
    used = dpth = 0;
    root = n - 2;
    next = n - 1;
    while (avbl > 0) {
        while (root >= 0 && (int)A[root].key == dpth) {
            used++;
            root--;
        }
        while (avbl > used) {
            A[next--].key = (uint32_t)dpth;
            avbl--;
        }
        avbl = 2 * used;
        dpth++;
        used = 0;
    }
}

/* Length-limited code lengths for freq[0..n). At least two symbols always get
 * a code so every table is complete, as strict inflaters require. */
static void huff_lengths(const uint32_t *freq_in, int n, int limit, uint8_t *lens)
{
    uint32_t freq[288];
    hsym s[288];
    int m = 0, count[64] = {0};

    memcpy(freq, freq_in, sizeof(uint32_t) * n);
    for (int i = 0; i < n; i++) m += freq[i] != 0;
    for (int i = 0; m < 2 && i < n; i++)
        if (!freq[i]) freq[i] = 1, m++;

    m = 0;
    for (int i = 0; i < n; i++)
        if (freq[i]) {
            s[m].key = freq[i];
            s[m].sym = (uint16_t)i;
            m++;
        }
    qsort(s, (size_t)m, sizeof *s, hsym_cmp);
    min_redundancy(s, m);
    for (int i = 0; i < m; i++) count[s[i].key > 63 ? 63 : s[i].key]++;

    /* Fold over-long codes into the limit and rebalance the Kraft sum. */
    for (int i = limit + 1; i < 64; i++) count[limit] += count[i], count[i] = 0;
    uint32_t total = 0;
    for (int i = limit; i > 0; i--) total += (uint32_t)count[i] << (limit - i);
    while (total != (1u << limit)) {
        count[limit]--;
        for (int i = limit - 1; i > 0; i--)
            if (count[i]) {
                count[i]--;
                count[i + 1] += 2;
                break;
            }
        total--;
    }

    memset(lens, 0, (size_t)n);
    for (int len = 1, j = m; len <= limit; len++)
        for (int k = count[len]; k > 0; k--) lens[s[--j].sym] = (uint8_t)len;
}

/* Canonical codes, bit-reversed because deflate packs bits LSB first. */
static void huff_codes(const uint8_t *lens, int n, uint16_t *codes)
{
    int count[16] = {0}, next[16] = {0}, code = 0;
    for (int i = 0; i < n; i++) count[lens[i]]++;
    count[0] = 0;
    for (int b = 1; b < 16; b++) {
        code = (code + count[b - 1]) << 1;
        next[b] = code;
    }
    for (int i = 0; i < n; i++) {
        int l = lens[i];
        unsigned c = l ? (unsigned)next[l]++ : 0, r = 0;
        for (int k = 0; k < l; k++, c >>= 1) r = r << 1 | (c & 1);
        codes[i] = (uint16_t)r;
    }
}

/* ---- bit writer -------------------------------------------------------- */

typedef struct {
    uint8_t *p;
    size_t n, cap;
    uint64_t bits;
    int nbits;
    int oom;
} bitw;

static int bw_reserve(bitw *w, size_t extra)
{
    if (w->n + extra <= w->cap) return 1;
    size_t cap = w->cap * 2;
    if (cap < w->n + extra + 4096) cap = w->n + extra + 4096;
    uint8_t *p = (uint8_t *)realloc(w->p, cap);
    if (!p) {
        w->oom = 1;
        return 0;
    }
    w->p = p;
    w->cap = cap;
    return 1;
}

static inline void bw_put(bitw *w, uint32_t v, int n)
{
    w->bits |= (uint64_t)v << w->nbits;
    w->nbits += n;
    if (w->nbits >= 32) {
        uint32_t lo = (uint32_t)w->bits;
        w->p[w->n] = (uint8_t)lo;
        w->p[w->n + 1] = (uint8_t)(lo >> 8);
        w->p[w->n + 2] = (uint8_t)(lo >> 16);
        w->p[w->n + 3] = (uint8_t)(lo >> 24);
        w->n += 4;
        w->bits >>= 32;
        w->nbits -= 32;
    }
}

static void bw_align(bitw *w)
{
    w->nbits = (w->nbits + 7) & ~7;
    while (w->nbits > 0) {
        w->p[w->n++] = (uint8_t)w->bits;
        w->bits >>= 8;
        w->nbits -= 8;
    }
    w->bits = 0;
    w->nbits = 0;
}

/* Stored block(s); n == 0 writes the empty block used as a sync flush. */
static void put_stored(bitw *w, const uint8_t *p, size_t n, int final)
{
    if (!bw_reserve(w, n + 5 * (n / 65535 + 1) + 16)) return;
    do {
        size_t k = n > 65535 ? 65535 : n;
        bw_put(w, final && k == n, 1);
        bw_put(w, 0, 2);
        bw_align(w);
        w->p[w->n++] = (uint8_t)k;
        w->p[w->n++] = (uint8_t)(k >> 8);
        w->p[w->n++] = (uint8_t)~k;
        w->p[w->n++] = (uint8_t)(~k >> 8);
        if (k) memcpy(w->p + w->n, p, k);
        w->n += k;
        p += k;
        n -= k;
    } while (n > 0);
}

/* ---- deflate ----------------------------------------------------------- */

#define WIN_BITS 15
#define WIN (1 << WIN_BITS)
#define WMASK (WIN - 1)
#define HASH_BITS 15
#define MIN_MATCH 4
#define MAX_MATCH 258
#define BLOCK_SYMS 32768

typedef struct {
    uint16_t ll; /* literal byte, or match length when d != 0 */
    uint16_t d;  /* match distance, 0 for a literal */
} sym;

typedef struct {
    int chain, lazy, nice;
} level_cfg;

static const level_cfg LEVELS[10] = {
    {0, 0, 0},     {1, 0, 32},     {4, 0, 64},     {8, 0, 128},    {16, 1, 128},
    {32, 1, 258},  {64, 1, 258},   {128, 1, 258},  {512, 1, 258},  {2048, 1, 258},
};

typedef struct {
    const ztables *t;
    const uint8_t *src;
    size_t n;
    bitw w;
    sym *syms;
    int nsyms;
    size_t block_start;
    uint32_t lfreq[286], dfreq[30];
    int32_t *head, *prev;
} dstate;

static uint64_t data_bits(const dstate *d, const uint8_t *ll_len, const uint8_t *d_len)
{
    uint64_t bits = 0;
    for (int i = 0; i < 286; i++) bits += (uint64_t)d->lfreq[i] * ll_len[i];
    for (int i = 257; i < 286; i++) bits += (uint64_t)d->lfreq[i] * len_extra[i - 257];
    for (int i = 0; i < 30; i++) bits += (uint64_t)d->dfreq[i] * (d_len[i] + dist_extra[i]);
    return bits;
}

static void put_symbols(const ztables *t, bitw *w, const sym *s, int ns, const uint8_t *ll_len,
                        const uint16_t *ll_code, const uint8_t *d_len, const uint16_t *d_code)
{
    for (int i = 0; i < ns; i++) {
        if (!s[i].d) {
            bw_put(w, ll_code[s[i].ll], ll_len[s[i].ll]);
        } else {
            int lc = t->len_code[s[i].ll], dc = dist_code(t, s[i].d);
            bw_put(w, ll_code[257 + lc], ll_len[257 + lc]);
            if (len_extra[lc]) bw_put(w, (uint32_t)(s[i].ll - len_base[lc]), len_extra[lc]);
            bw_put(w, d_code[dc], d_len[dc]);
            if (dist_extra[dc]) bw_put(w, (uint32_t)(s[i].d - dist_base[dc]), dist_extra[dc]);
        }
    }
    bw_put(w, ll_code[256], ll_len[256]);
}

static int rle_lengths(const uint8_t *lens, int n, uint8_t *sy, uint8_t *ex)
{
    int m = 0, i = 0;
    while (i < n) {
        int v = lens[i], run = 1;
        while (i + run < n && lens[i + run] == v) run++;
        i += run;
        if (v == 0) {
            while (run >= 11) {
                int r = run > 138 ? 138 : run;
                sy[m] = 18, ex[m++] = (uint8_t)(r - 11), run -= r;
            }
            if (run >= 3) sy[m] = 17, ex[m++] = (uint8_t)(run - 3), run = 0;
        } else {
            sy[m] = (uint8_t)v, ex[m++] = 0, run--;
            while (run >= 3) {
                int r = run > 6 ? 6 : run;
                sy[m] = 16, ex[m++] = (uint8_t)(r - 3), run -= r;
            }
        }
        while (run-- > 0) sy[m] = (uint8_t)v, ex[m++] = 0;
    }
    return m;
}

/* Emit the pending symbols covering src[block_start, end) as the cheapest of a
 * dynamic, fixed or stored block. */
static void flush_block(dstate *d, size_t end, int final)
{
    bitw *w = &d->w;
    uint8_t ll_len[286], d_len[30], cl_len[19], lens[316], sy[316], ex[316];
    uint16_t ll_code[286], d_code[30], cl_code[19];
    uint32_t clfreq[19] = {0};

    d->lfreq[256] = 1;
    huff_lengths(d->lfreq, 286, 15, ll_len);
    huff_lengths(d->dfreq, 30, 15, d_len);
    int hlit = 286, hdist = 30;
    while (hlit > 257 && !ll_len[hlit - 1]) hlit--;
    while (hdist > 1 && !d_len[hdist - 1]) hdist--;
    memcpy(lens, ll_len, (size_t)hlit);
    memcpy(lens + hlit, d_len, (size_t)hdist);
    int nrle = rle_lengths(lens, hlit + hdist, sy, ex);
    for (int i = 0; i < nrle; i++) clfreq[sy[i]]++;
    huff_lengths(clfreq, 19, 7, cl_len);
    int hclen = 19;
    while (hclen > 4 && !cl_len[cl_order[hclen - 1]]) hclen--;

    uint64_t dyn = 17 + 3 * (uint64_t)hclen + data_bits(d, ll_len, d_len);
    for (int i = 0; i < nrle; i++) dyn += cl_len[sy[i]] + (sy[i] == 16 ? 2 : sy[i] == 17 ? 3 : sy[i] == 18 ? 7 : 0);
    const ztables *t = d->t;
    uint64_t fix = 3 + data_bits(d, t->fixed_ll_len, t->fixed_d_len);
    size_t raw = end - d->block_start;
    uint64_t sto = ((uint64_t)raw + 5 * (raw / 65535 + 1)) * 8 + 8;

    if (sto <= dyn && sto <= fix) {
        put_stored(w, d->src + d->block_start, raw, final);
    } else if (bw_reserve(w, (size_t)((dyn < fix ? dyn : fix) / 8) + 64)) {
        if (dyn < fix) {
            huff_codes(ll_len, 286, ll_code);
            huff_codes(d_len, 30, d_code);
            huff_codes(cl_len, 19, cl_code);
            bw_put(w, (uint32_t)final, 1);
            bw_put(w, 2, 2);
            bw_put(w, (uint32_t)(hlit - 257), 5);
            bw_put(w, (uint32_t)(hdist - 1), 5);
            bw_put(w, (uint32_t)(hclen - 4), 4);
            for (int i = 0; i < hclen; i++) bw_put(w, cl_len[cl_order[i]], 3);
            for (int i = 0; i < nrle; i++) {
                bw_put(w, cl_code[sy[i]], cl_len[sy[i]]);
                if (sy[i] >= 16) bw_put(w, ex[i], sy[i] == 16 ? 2 : sy[i] == 17 ? 3 : 7);
            }
            put_symbols(t, w, d->syms, d->nsyms, ll_len, ll_code, d_len, d_code);
        } else {
            bw_put(w, (uint32_t)final, 1);
            bw_put(w, 1, 2);
            put_symbols(t, w, d->syms, d->nsyms, t->fixed_ll_len, t->fixed_ll_code, t->fixed_d_len,
                        t->fixed_d_code);
        }
    }
    memset(d->lfreq, 0, sizeof d->lfreq);
    memset(d->dfreq, 0, sizeof d->dfreq);
    d->nsyms = 0;
    d->block_start = end;
}

static inline uint32_t hash4(const uint8_t *p)
{
    uint32_t v;
    memcpy(&v, p, 4);
    return (v * 2654435761u) >> (32 - HASH_BITS);
}

static inline void insert(dstate *d, size_t pos)
{
    if (pos + MIN_MATCH > d->n) return;
    uint32_t h = hash4(d->src + pos);
    d->prev[pos & WMASK] = d->head[h];
    d->head[h] = (int32_t)pos;
}

static inline int match_len(const uint8_t *a, const uint8_t *b, int maxlen)
{
    int l = 0;
#if !defined(__BYTE_ORDER__) || __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    while (l + 8 <= maxlen) {
        uint64_t x, y;
        memcpy(&x, a + l, 8);
        memcpy(&y, b + l, 8);
        if (x != y) return l + (ctz64(x ^ y) >> 3);
        l += 8;
    }
#endif
    while (l < maxlen && a[l] == b[l]) l++;
    return l;
}

/* Longest match for pos, walking the hash chain of an already inserted pos. */
static int find_match(const dstate *d, size_t pos, int chain, int nice, int *dist)
{
    const uint8_t *src = d->src;
    int maxlen = d->n - pos > MAX_MATCH ? MAX_MATCH : (int)(d->n - pos);
    if (maxlen < MIN_MATCH) return 0;
    int best = MIN_MATCH - 1;
    int64_t lowest = (int64_t)pos - WIN;
    int32_t c = d->prev[pos & WMASK];
    uint32_t want;
    memcpy(&want, src + pos, 4);
    while (c >= 0 && c > lowest && chain-- > 0) {
        const uint8_t *a = src + c, *b = src + pos;
        uint32_t have;
        memcpy(&have, a, 4);
        if (a[best] == b[best] && have == want) {
            int l = match_len(a, b, maxlen);
            if (l > best) {
                best = l;
                *dist = (int)(pos - (size_t)c);
                if (l >= nice || l == maxlen) break;
            }
        }
        int32_t nc = d->prev[c & WMASK];
        if (nc >= c) break;
        c = nc;
    }
    return best >= MIN_MATCH ? best : 0;
}

static inline void put_lit(dstate *d, int c)
{
    d->syms[d->nsyms].ll = (uint16_t)c;
    d->syms[d->nsyms++].d = 0;
    d->lfreq[c]++;
}

static inline void put_match(dstate *d, int len, int dist)
{
    d->syms[d->nsyms].ll = (uint16_t)len;
    d->syms[d->nsyms++].d = (uint16_t)dist;
    d->lfreq[257 + d->t->len_code[len]]++;
    d->dfreq[dist_code(d->t, dist)]++;
}

/* Compress src into raw deflate blocks. A non-final band ends with an empty
 * stored block so the output is byte aligned and can be concatenated. */
static void deflate_band(dstate *d, int level, int final)
{
    if (level <= 0) {
        put_stored(&d->w, d->src, d->n, final);
        return;
    }
    const level_cfg cfg = LEVELS[level > 9 ? 9 : level];
    const uint8_t *src = d->src;
    const size_t n = d->n;
    for (int i = 0; i < (1 << HASH_BITS); i++) d->head[i] = -1;

    size_t i = 0;
    while (i < n) {
        int dist = 0, len = 0;
        insert(d, i);
        len = find_match(d, i, cfg.chain, cfg.nice, &dist);
        if (len && cfg.lazy && len < cfg.nice) {
            int dist2 = 0;
            insert(d, i + 1);
            size_t next = i + 2; /* first position not yet in the hash chains */
            int len2 = find_match(d, i + 1, cfg.chain, cfg.nice, &dist2);
            if (len2 > len) {
                put_lit(d, src[i]);
                i++;
                len = len2;
                dist = dist2;
            }
            put_match(d, len, dist);
            for (size_t j = next; j < i + (size_t)len; j++) insert(d, j);
            i += (size_t)len;
        } else if (len) {
            put_match(d, len, dist);
            for (size_t j = i + 1; j < i + (size_t)len; j++) insert(d, j);
            i += (size_t)len;
        } else {
            put_lit(d, src[i]);
            i++;
        }
        if (d->nsyms >= BLOCK_SYMS - 2) flush_block(d, i, 0);
        if (d->w.oom) return;
    }
    flush_block(d, n, final);
    if (!final) put_stored(&d->w, NULL, 0, 0);
    else if (bw_reserve(&d->w, 8)) bw_align(&d->w);
}

/* ---- PNG --------------------------------------------------------------- */

typedef struct {
    const ztables *t;
    const ie_image *img;
    int ch, level, nbands, rows_per_band;
    uint8_t **out;
    size_t *out_len;
    uint32_t *adler;
    size_t *raw_len;
    volatile int failed;
} png_job;

static inline int paeth(int a, int b, int c)
{
    int p = a + b - c, pa = abs(p - a), pb = abs(p - b), pc = abs(p - c);
    return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
}

static void pack_row(const uint8_t *s, uint8_t *d, int w, int ch)
{
    switch (ch) {
    case 4: memcpy(d, s, (size_t)w * 4); break;
    case 3:
        for (int x = 0; x < w; x++, s += 4, d += 3) d[0] = s[0], d[1] = s[1], d[2] = s[2];
        break;
    case 2:
        for (int x = 0; x < w; x++, s += 4, d += 2) d[0] = s[0], d[1] = s[3];
        break;
    default:
        for (int x = 0; x < w; x++, s += 4) *d++ = s[0];
    }
}

/* Filter rows [y0, y1) choosing per row the filter with the smallest sum of
 * absolute signed residuals (the usual libpng heuristic). */
static void filter_rows(const ie_image *img, int ch, int y0, int y1, uint8_t *out, uint8_t *scratch)
{
    const int w = img->w;
    const size_t stride = (size_t)w * ch;
    uint8_t *prv = scratch, *cur = scratch + stride, *f = scratch + 2 * stride;
    if (y0 > 0) pack_row(img->px + (size_t)(y0 - 1) * w * 4, prv, w, ch);
    else memset(prv, 0, stride);

    for (int y = y0; y < y1; y++) {
        pack_row(img->px + (size_t)y * w * 4, cur, w, ch);
        uint32_t sum[5] = {0, 0, 0, 0, 0};
        for (size_t i = 0; i < stride; i++) {
            int x = cur[i], a = i >= (size_t)ch ? cur[i - ch] : 0, b = prv[i], c = i >= (size_t)ch ? prv[i - ch] : 0;
            uint8_t v1 = (uint8_t)(x - a), v2 = (uint8_t)(x - b), v3 = (uint8_t)(x - ((a + b) >> 1)),
                    v4 = (uint8_t)(x - paeth(a, b, c));
            f[i] = v4;
            sum[0] += x < 128 ? x : 256 - x;
            sum[1] += v1 < 128 ? v1 : 256 - v1;
            sum[2] += v2 < 128 ? v2 : 256 - v2;
            sum[3] += v3 < 128 ? v3 : 256 - v3;
            sum[4] += v4 < 128 ? v4 : 256 - v4;
        }
        int best = 0;
        for (int k = 1; k < 5; k++)
            if (sum[k] < sum[best]) best = k;
        *out++ = (uint8_t)best;
        if (best == 4) {
            memcpy(out, f, stride);
        } else {
            for (size_t i = 0; i < stride; i++) {
                int x = cur[i], a = i >= (size_t)ch ? cur[i - ch] : 0, b = prv[i];
                out[i] = (uint8_t)(best == 0 ? x : best == 1 ? x - a : best == 2 ? x - b : x - ((a + b) >> 1));
            }
        }
        out += stride;
        uint8_t *t = prv;
        prv = cur;
        cur = t;
    }
}

static void put_be32(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)(v >> 24);
    p[1] = (uint8_t)(v >> 16);
    p[2] = (uint8_t)(v >> 8);
    p[3] = (uint8_t)v;
}

static void png_band(void *ctx, int b0, int b1)
{
    png_job *j = (png_job *)ctx;
    const size_t stride = (size_t)j->img->w * j->ch;
    for (int b = b0; b < b1 && !j->failed; b++) {
        int y0 = b * j->rows_per_band;
        int y1 = y0 + j->rows_per_band > j->img->h ? j->img->h : y0 + j->rows_per_band;
        size_t raw = (size_t)(y1 - y0) * (stride + 1);
        uint8_t *filt = (uint8_t *)malloc(raw + 3 * stride);
        dstate d;
        memset(&d, 0, sizeof d);
        d.head = (int32_t *)malloc(sizeof(int32_t) << HASH_BITS);
        d.prev = (int32_t *)malloc(sizeof(int32_t) * WIN);
        d.syms = (sym *)malloc(sizeof(sym) * BLOCK_SYMS);
        if (!filt || !d.head || !d.prev || !d.syms || !bw_reserve(&d.w, raw / 2 + 1024)) {
            j->failed = 1;
        } else {
            filter_rows(j->img, j->ch, y0, y1, filt, filt + raw);
            j->adler[b] = adler32(filt, raw);
            j->raw_len[b] = raw;
            /* 8-byte IDAT header first; band 0 also carries the zlib header. */
            d.w.n = 8;
            if (b == 0) {
                d.w.p[d.w.n++] = 0x78;
                d.w.p[d.w.n++] = 0x9C;
            }
            d.t = j->t;
            d.src = filt;
            d.n = raw;
            deflate_band(&d, j->level, b == j->nbands - 1);
            if (d.w.oom || !bw_reserve(&d.w, 4)) {
                j->failed = 1;
            } else {
                size_t data_len = d.w.n - 8;
                put_be32(d.w.p, (uint32_t)data_len);
                memcpy(d.w.p + 4, "IDAT", 4);
                put_be32(d.w.p + d.w.n, crc32_update(j->t->crc, 0, d.w.p + 4, data_len + 4));
                d.w.n += 4;
                j->out[b] = d.w.p;
                j->out_len[b] = d.w.n;
                d.w.p = NULL;
            }
        }
        free(filt);
        free(d.head);
        free(d.prev);
        free(d.syms);
        free(d.w.p);
    }
}

static void put_chunk(const ztables *t, uint8_t **p, const char *type, const uint8_t *data, uint32_t len)
{
    put_be32(*p, len);
    memcpy(*p + 4, type, 4);
    if (len) memcpy(*p + 8, data, len);
    put_be32(*p + 8 + len, crc32_update(t->crc, 0, *p + 4, len + 4));
    *p += 12 + len;
}

int ie_png_encode(const ie_image *img, int level, uint8_t **out, size_t *out_len)
{
    *out = NULL;
    *out_len = 0;
    if (!img || !img->px || img->w <= 0 || img->h <= 0) return IE_EINVAL;
    ztables *t = (ztables *)malloc(sizeof *t);
    if (!t) return IE_ENOMEM;
    tables_init(t);

    int alpha = ie_has_alpha(img), gray = ie_is_gray(img);
    int ch = gray ? (alpha ? 2 : 1) : (alpha ? 4 : 3);
    int color_type = gray ? (alpha ? 4 : 0) : (alpha ? 6 : 2);
    size_t stride = (size_t)img->w * ch + 1;
    size_t total = stride * img->h;

    int threads = ie_get_threads();
    int nbands = (int)((total + (512u << 10) - 1) / (512u << 10));
    if (nbands > threads * 2) nbands = threads * 2;
    /* Keep bands well below 2 GB so positions and chunk lengths fit in 31 bits. */
    if ((size_t)nbands < total / (256u << 20) + 1) nbands = (int)(total / (256u << 20) + 1);
    if (nbands > img->h) nbands = img->h;
    if (nbands < 1) nbands = 1;
    int rpb = (img->h + nbands - 1) / nbands;
    nbands = (img->h + rpb - 1) / rpb;

    png_job j;
    memset(&j, 0, sizeof j);
    j.t = t;
    j.img = img;
    j.ch = ch;
    j.level = level < 0 ? 6 : level > 9 ? 9 : level;
    j.nbands = nbands;
    j.rows_per_band = rpb;
    j.out = (uint8_t **)calloc((size_t)nbands, sizeof(uint8_t *));
    j.out_len = (size_t *)calloc((size_t)nbands, sizeof(size_t));
    j.adler = (uint32_t *)calloc((size_t)nbands, sizeof(uint32_t));
    j.raw_len = (size_t *)calloc((size_t)nbands, sizeof(size_t));
    int err = IE_ENOMEM;
    if (j.out && j.out_len && j.adler && j.raw_len) {
        ie_parallel_for(nbands, 1, png_band, &j);
        if (!j.failed) {
            uint32_t adler = j.adler[0];
            size_t size = 8 + 25 + 16 + 12;
            for (int b = 0; b < nbands; b++) size += j.out_len[b];
            for (int b = 1; b < nbands; b++) adler = adler32_combine(adler, j.adler[b], j.raw_len[b]);
            uint8_t *png = (uint8_t *)malloc(size);
            if (png) {
                static const uint8_t sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
                uint8_t ihdr[13], tail[4], *p = png;
                memcpy(p, sig, 8);
                p += 8;
                put_be32(ihdr, (uint32_t)img->w);
                put_be32(ihdr + 4, (uint32_t)img->h);
                ihdr[8] = 8;
                ihdr[9] = (uint8_t)color_type;
                ihdr[10] = ihdr[11] = ihdr[12] = 0;
                put_chunk(t, &p, "IHDR", ihdr, 13);
                for (int b = 0; b < nbands; b++) {
                    memcpy(p, j.out[b], j.out_len[b]);
                    p += j.out_len[b];
                }
                put_be32(tail, adler);
                put_chunk(t, &p, "IDAT", tail, 4);
                put_chunk(t, &p, "IEND", NULL, 0);
                *out = png;
                *out_len = (size_t)(p - png);
                err = IE_OK;
            }
        }
    }
    if (j.out)
        for (int b = 0; b < nbands; b++) free(j.out[b]);
    free(j.out);
    free(j.out_len);
    free(j.adler);
    free(j.raw_len);
    free(t);
    return err;
}
