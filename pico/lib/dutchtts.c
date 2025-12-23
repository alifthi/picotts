#include <dutchtts.h>

// Session initialization
int init_session(DutchTTSContext * ctx, const char * text2mel_path,
                const char * vocoder_path, const char * config_path){
    
    ctx->ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

    ctx->env = NULL;
    ctx->sess_opts = NULL;
    ctx->meminfo = NULL;

    ctx->ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "tts", &env);
    ctx->ort->CreateSessionOptions(&sess_opts);
    ctx->ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &meminfo);
    
    ctx->text2mel_session = NULL;
    ctx->vocoder_session = NULL;

    ctx->ort->CreateSession(ctx->env, text2mel_path, ctx->sess_opts, &ctx->text2mel_session);
    if (!ctx->text2mel_session) { 
        fprintf(stderr, "Failed to create text2mel session\n"); 
        ctx->ort->ReleaseMemoryInfo(ctx->meminfo);
        ctx->ort->ReleaseSessionOptions(ctx->sess_opts);
        ctx->ort->ReleaseEnv(ctx->env);
        return 1; 
    }
    ctx->ort->CreateSession(ctx->env, vocoder_path, ctx->sess_opts, &ctx->vocoder_session);

    if (!ctx->vocoder_session) {
        fprintf(stderr, "Failed to create vocoder session\n");
        ctx->ort->ReleaseSession(ctx->text2mel_session);
        ctx->ort->ReleaseMemoryInfo(ctx->meminfo);
        ctx->ort->ReleaseSessionOptions(ctx->sess_opts);
        ctx->ort->ReleaseEnv(ctx->env);
        return 1;
    }
    return 0;
}

// Inference from mel2text model.
int run_mel2text_session(DutchTTSContext * ctx, IntVec ids,
                        OrtValue* mel_tensor){
    
    int64_t input_ids[ids.len];       
    int64_t input_ids_shape[] = {1, ids.len};
    int64_t input_len[] = {ids.len};
    int64_t input_len_shape[] = {1};

    for (size_t i = 0; i < ids.len; ++i) {
        input_ids[i] = (int64_t)ids.data[i];
    }


    OrtValue* ids_tensor = NULL;
    OrtValue* len_tensor = NULL;

    ctx->ort->CreateTensorWithDataAsOrtValue(meminfo, input_ids, sizeof(input_ids),
                                         input_ids_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &ids_tensor);
    
    ctx->ort->CreateTensorWithDataAsOrtValue(meminfo, input_len, sizeof(input_len),
                                         input_len_shape, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &len_tensor);

    const char* glow_in_names[] = {"input1", "input2"};
    const OrtValue* glow_in_vals[] = {ids_tensor, len_tensor};
    const char* glow_out_names[] = {"output"};
    mel_tensor = NULL;

    ctx->ort->Run(glow_sess, NULL, glow_in_names, glow_in_vals, 2, glow_out_names, 1, &mel_tensor);
    if (!mel_tensor) { 
        fprintf(stderr, "GlowTTS Run failed or returned NULL mel\n"); 
        ctx->ort->ReleaseSession(ctx->text2mel_session);
        ctx->ort->ReleaseSession(ctx->vocoder_session);
        ctx->ort->ReleaseMemoryInfo(ctx->meminfo);
        ctx->ort->ReleaseSessionOptions(ctx->sess_opts);
        ctx->ort->ReleaseEnv(ctx->env);
        ctx->ort->ReleaseValue(ids_tensor);
        ctx->ort->ReleaseValue(len_tensor);
        ctx->ort->ReleaseValue(glow_in_vals);
        free(glow_in_names);
        free(glow_out_names);

        return 1; 
    }
    ctx->ort->ReleaseValue(ids_tensor);
    ctx->ort->ReleaseValue(len_tensor);
    ctx->ort->ReleaseValue(glow_in_vals);
    free(glow_in_names);
    free(glow_out_names);
    return 0;

}

int remove_zeros(DutchTTSContext * ctx, OrtValue* mel_tensor,
                OrtValue* mel_trim_tensor){
    
    float* mel_data = NULL;
    ctx->ort->GetTensorMutableData(mel_tensor, (void**)&mel_data);

    OrtTensorTypeAndShapeInfo* mel_info = NULL;
    ctx->ort->GetTensorTypeAndShape(mel_tensor, &mel_info);

    int64_t dims[3] = {0,0,0};
    ctx->ort->GetDimensions(mel_info, dims, 3);
    printf("Glow mel shape: [%lld, %lld, %lld]\n", (long long)dims[0], (long long)dims[1], (long long)dims[2]);

    int layout = -1; 
    int channels = 0;
    int frames = 0;

    if (dims[0] != 1) {
        fprintf(stderr, "Warning: expected batch dim 1, got %lld\n", (long long)dims[0]);
    }

    if (dims[1] == 80) {
        layout = 0;
        channels = (int)dims[1];
        frames = (int)dims[2];
    } else if (dims[2] == 80) {
        layout = 1;
        channels = (int)dims[2];
        frames = (int)dims[1];
    } else {
        if (dims[1] == 80 || dims[2] == 80) {
            layout = (dims[1] == 80) ? 0 : 1;
            channels = 80;
            frames = (layout==0) ? (int)dims[2] : (int)dims[1];
        } else {
            fprintf(stderr, "Unexpected mel channels dimension (neither dims[1] nor dims[2] == 80). Using dims[1] as channels.\n");
            layout = 0;
            channels = (int)dims[1];
            frames = (int)dims[2];
        }
    }
    int last_nonzero = -1;
    for (int f = frames - 1; f >= 0; --f) {
        int any = 0;
        for (int c = 0; c < channels; ++c) {
            size_t idx = 0;
            if (layout == 0) {
                idx = (size_t)c * (size_t)frames + (size_t)f;
            } else {
                idx = (size_t)f * (size_t)channels + (size_t)c;
            }
            float v = mel_data[idx];
            if (v > EPS || v < -EPS) { any = 1; break; }
        }
        if (any) { last_nonzero = f; break; }
    }

    if (last_nonzero < 0) {
        fprintf(stderr, "All mel frames are (near) zero. Exiting.\n");
        ctx->ort->ReleaseTensorTypeAndShapeInfo(mel_info);
        ctx->ort->ReleaseValue(mel_tensor);
        ctx->ort->ReleaseSession(ctx->text2mel_session);
        ctx->ort->ReleaseSession(ctx->vocoder_session);
        ctx->ort->ReleaseMemoryInfo(ctx->meminfo);
        ctx->ort->ReleaseSessionOptions(ctx->sess_opts);
        ctx->ort->ReleaseEnv(ctx->env);
        return 1;
    }
     int new_frames = last_nonzero + 1;
    
    printf("Trimming frames: old=%d new=%d (last nonzero frame=%d)\n", frames, new_frames, last_nonzero);

    int64_t new_dims[3];
    if (layout == 0) {
        new_dims[0] = 1;
        new_dims[1] = channels;
        new_dims[2] = new_frames;
    } else {
        new_dims[0] = 1;
        new_dims[1] = new_frames;
        new_dims[2] = channels;
    }

    OrtAllocator* allocator = NULL;
    ctx->ort->GetAllocatorWithDefaultOptions(&allocator);
    ctx->ort->CreateTensorAsOrtValue(allocator, new_dims, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &mel_trim_tensor);
    if (!mel_trim_tensor) {
        fprintf(stderr, "CreateTensorAsOrtValue failed\n");
        ctx->ort->ReleaseTensorTypeAndShapeInfo(mel_info);
        ctx->ort->ReleaseValue(mel_tensor);
        ctx->ort->ReleaseValue(ids_tensor);
        ctx->ort->ReleaseValue(len_tensor);
        ctx->ort->ReleaseSession(ctx->glow_sess);
        ctx->ort->ReleaseSession(ctx->text2mel_session);
        ctx->ort->ReleaseMemoryInfo(ctx->meminfo);
        ctx->ort->ReleaseSessionOptions(ctx->sess_opts);
        ctx->ort->ReleaseEnv(ctx->env);
        ctx->ort->ReleaseAllocator(allocator)
        free(mel_data);
        return 1;
    }

    float* mel_trim_data = NULL;
    ctx->ort->GetTensorMutableData(mel_trim_tensor, (void**)&mel_trim_data);

    for (int f = 0; f < new_frames; ++f) {
        for (int c = 0; c < channels; ++c) {
            size_t src = (layout == 0) ? (size_t)c * frames + f : (size_t)f * channels + c;
            size_t dst = (layout == 0) ? (size_t)c * new_frames + f : (size_t)f * channels + c;
            mel_trim_data[dst] = mel_data[src];
        }
    }
    ctx->ort->ReleaseAllocator(allocator)
    free(mel_data);
    free(mel_trim_data);
    return 0;

}

int run_vocoder_session(DutchTTSContext * ctx,
                        OrtValue* mel_trim_tensor,
                        float* audio_data){
    const char* voc_in_names[] = {"input1"};
    const OrtValue* voc_in_vals[] = {mel_trim_tensor};
    const char* voc_out_names[] = {"output"};
    OrtValue* audio_tensor = NULL;

    ctx->ort->Run(vocoder_sess, NULL, voc_in_names, voc_in_vals, 1, voc_out_names, 1, &audio_tensor);
    if (!audio_tensor) {
        fprintf(stderr, "Vocoder run failed or returned NULL audio tensor\n");

        ctx->ort->ReleaseValue(mel_trim_tensor);
        ctx->ort->ReleaseValue(mel_tensor);
        ctx->ort->ReleaseValue(ids_tensor);
        ctx->ort->ReleaseValue(len_tensor);
        ctx->ort->ReleaseSession(ctx->vocoder_sess);
        ctx->ort->ReleaseSession(ctx->glow_sess);
        ctx->ort->ReleaseMemoryInfo(ctx->meminfo);
        ctx->ort->ReleaseSessionOptions(ctx->sess_opts);
        ctx->ort->ReleaseEnv(ctx->env);

        return 1;
    }

    ctx->ort->GetTensorMutableData(audio_tensor, (void**)&audio_data);

    OrtTensorTypeAndShapeInfo* audio_info = NULL;
    ctx->ort->GetTensorTypeAndShape(audio_tensor, &audio_info);

    size_t audio_len = 0;
    ctx->ort->GetTensorShapeElementCount(audio_info, &audio_len);


    ctx->ort->ReleaseTensorTypeAndShapeInfo(audio_info);
    ctx->ort->ReleaseValue(audio_tensor);
    ctx->ort->ReleaseValue(mel_trim_tensor);
    ctx->ort->ReleaseValue(mel_tensor);
    ctx->ort->ReleaseValue(ids_tensor);
    ctx->ort->ReleaseValue(len_tensor);
    ctx->ort->ReleaseSession(ctx->vocoder_sess);
    ctx->ort->ReleaseSession(ctx->glow_sess);
    ctx->ort->ReleaseMemoryInfo(ctx->meminfo);
    ctx->ort->ReleaseSessionOptions(ctx->sess_opts);
    ctx->ort->ReleaseEnv(ctx->env);

    return 0;
}