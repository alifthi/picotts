#ifndef DUTCH_IPA_H
#define DUTCH_IPA_H

#include <stdlib.h>
#include <espeak-ng/speak_lib.h>

extern int TEXT_MODE;                  
extern int PHONEM_MODE;  

int init_dutch_ipa();

int text_to_phonemes(const char *text, char * ipas);

#endif