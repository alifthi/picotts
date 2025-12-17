#ifndef DUTCH_IPA_H
#define DUTCH_IPA_H

#include <stdlib.h>
#include <espeak-ng/speak_lib.h>

int TEXT_MODE = 0;                  
int PHONEM_MODE = espeakPHONEMES_IPA;  

int init_dutch_ipa();

int text_to_phonemes(const char *text, const void **textptr, char * ipas);

#endif