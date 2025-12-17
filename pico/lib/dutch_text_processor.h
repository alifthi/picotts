#ifndef TEXT_PROCESSOR_H
#define TEXT_PROCESSOR_H

#include <stddef.h>

// IntVec value type to store the ids
typedef struct {
    int *data;
    size_t len;
    size_t cap;
} IntVec;

// A structure to store the id-symbol pairs
typedef struct {
    const char *sym;
    int id;
    size_t len;
} SymbolEntry;


// Symbol-id mapp
static SymbolEntry SYMBOLS_INIT[] = {
    {"_", 0, 0}, {"|", 1, 0}, {"‖", 2, 0}, {"#", 3, 0},
    {"a", 4, 0}, {"aː", 5, 0}, {"b", 6, 0}, {"c", 7, 0}, {"d", 8, 0},
    {"e", 9, 0}, {"eː", 10, 0}, {"f", 11, 0}, {"h", 12, 0},
    {"i", 13, 0}, {"iː", 14, 0}, {"j", 15, 0}, {"k", 16, 0},
    {"l", 17, 0}, {"m", 18, 0}, {"n", 19, 0}, {"o", 20, 0},
    {"p", 21, 0}, {"s", 22, 0}, {"t", 23, 0}, {"u", 24, 0},
    {"uː", 25, 0}, {"v", 26, 0}, {"w", 27, 0}, {"x", 28, 0},
    {"y", 29, 0}, {"yː", 30, 0}, {"z", 31, 0}, {"ø", 32, 0},
    {"ŋ", 33, 0}, {"œy", 34, 0}, {"œː", 35, 0}, {"ɑ", 36, 0},
    {"ɑu", 37, 0}, {"ɑː", 38, 0}, {"ɔ", 39, 0}, {"ɔː", 40, 0},
    {"ə", 41, 0}, {"ɛ", 42, 0}, {"ɛi", 43, 0}, {"ɛː", 44, 0},
    {"ɡ", 45, 0}, {"ɣ", 46, 0}, {"ɱ", 47, 0}, {"ɹ", 48, 0},
    {"ʃ", 49, 0}, {"ʏ", 50, 0}, {"ʏː", 51, 0},
    {"ʒ", 52, 0}, {"ʔ", 53, 0},
    {" ", 3, 0},
    {"eos", 0, 0},
};

static const char *NORMALIZE_SRC[] = {
    "ɪː", "ɪ", "r", "oː"
};
static const char *NORMALIZE_DST[] = {
    "iː", "i", "ɹ", "o"
};


static const size_t NORMALIZE_COUNT = sizeof(NORMALIZE_SRC) / sizeof(NORMALIZE_SRC[0]);


static void iv_init(IntVec *v);

static void iv_push(IntVec *v, int x);

static void iv_free(IntVec *v);

static int cmp_symbol_len_desc(const void *a, const void *b);

static char *replace_all(const char *s, const char *old, const char *newstr);

static char *apply_normalize(const char *ipa);

#endif 