#ifndef TTS_MODEL_H
#define TTS_MODEL_H

#include <stddef.h>
#include <tensorflow/lite/c/c_api.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float energy_ratio;
    int speaker_id;
    float f0_ratio;
    float speed_ratio;
} TTSConfig;

typedef struct {
    TfLiteInterpreter* tacotron2_interpreter;
    TfLiteInterpreter* melgan_interpreter;
    char* processor_path;
    TTSConfig config;
} TTSContext;

TTSContext* tts_initialize(const char* tacotron_model_path,
                         const char* melgan_model_path,
                         const char* processor_path);

void tts_configure(TTSContext* ctx, const TTSConfig* config);

int tts_generate_audio(TTSContext* ctx,
                      const char* text,
                      float** audio_out,
                      size_t* audio_length);

void tts_cleanup(TTSContext* ctx);

#ifdef __cplusplus
}
#endif

#endif