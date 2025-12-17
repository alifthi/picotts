#ifndef DUTCH_IPA_H
#define DUTCH_IPA_H

#include <stdlib.h>
#include <espeak-ng/speak_lib.h>

int init_dutch_ipa();

int text_to_phonemes(const void **textptr);

#endif