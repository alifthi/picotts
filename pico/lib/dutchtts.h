#ifndef DUTCHTTS_H
#define DUTCHTTS_H

#include <stddef.h>
#include <dutch_ipa.h>
#include <dutch_text_processor.h>
#include <onnxruntime_c_api.h>


#define MEL_EPS 1e-6f

typedef struct {
    OrtSession* text2mel_session;
    OrtSession* vocoder_session;
    char* processor_path;
    const OrtApi* ort;
    OrtSessionOptions* sess_opts;
    OrtMemoryInfo* meminfo;
} DutchTTSContext;

int init_session(DutchTTSContext * ctx,
     const char * text2mel_path,
     const char * vocoder_path,
     const char * config_path);

int run_mel2text_session(DutchTTSContext * ctx,
     IntVec ids,
     OrtValue* mel_tensor);

int renove_zeros(DutchTTSContext * ctx,
                OrtValue* mel_tensor,
                float* mel_data);

int run_vocoder_session(DutchTTSContext * ctx,
     float* mel_data,
     float* audio_data);
    

#endif