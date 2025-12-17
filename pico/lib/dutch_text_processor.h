#ifndef TEXT_PROCESSOR_H
#define TEXT_PROCESSOR_H

#include <stddef.h>
#include "text_symbols.h"


typedef struct {
    int *data;
    size_t len;
    size_t cap;
} IntVec;

static void iv_init(IntVec *v);

static void iv_push(IntVec *v, int x);

static void iv_free(IntVec *v);

static int cmp_symbol_len_desc(const void *a, const void *b);

static char *replace_all(const char *s, const char *old, const char *newstr);

static char *apply_normalize(const char *ipa);

#endif 