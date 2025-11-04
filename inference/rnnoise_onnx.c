#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <assert.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ONNX Runtime includes (C API)
#include <onnxruntime_c_api.h>

// RNNoise includes
#include "rnnoise.h"
#include "rnn.h"
#include "rnn_data.h"
#include "pitch.h"
#include "kiss_fft.h"
#include "common.h"
#include "opus_types.h"

// Audio processing constants
#define FRAME_SIZE 480  // 10ms at 48kHz
#define SAMPLE_RATE 48000
#define NB_FEATURES 42
#define NB_BANDS 22

// Additional constants from denoise.c
#define FREQ_SIZE (FRAME_SIZE + 1)
#define WINDOW_SIZE (2*FRAME_SIZE)
#define FRAME_SIZE_SHIFT 2
#define SQUARE(x) ((x)*(x))
#define MAX16(a,b) ((a) > (b) ? (a) : (b))
#define MIN16(a,b) ((a) < (b) ? (a) : (b))

// Band boundaries (same as in denoise.c)
static const opus_int16 eband5ms[] = {
/*0  200 400 600 800  1k 1.2 1.4 1.6  2k 2.4 2.8 3.2  4k 4.8 5.6 6.8  8k 9.6 12k 15.6 20k*/
  0,  1,  2,  3,  4,  5,  6,  7,  8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100
};

// WAV file header structure
typedef struct {
    char riff[4];
    uint32_t file_size;
    char wave[4];
    char fmt[4];
    uint32_t fmt_size;
    uint16_t audio_format;
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
    char data[4];
    uint32_t data_size;
} WAVHeader;

// Audio processing context
typedef struct {
    // ONNX Runtime
    const OrtApi* api;
    OrtEnv* env;
    OrtSession* session;
    OrtSessionOptions* session_options;
    OrtAllocator* allocator;
    
    // Input/Output names
    char* input_name;
    char* input_name_vad_state;
    char* input_name_noise_state;
    char* input_name_denoise_state;
    char* output_name_denoise;
    char* output_name_vad;
    char* output_name_vad_state;
    char* output_name_noise_state;
    char* output_name_denoise_state;
    
    // Model type detection
    int has_gru_states;  // 1 if model has GRU state inputs/outputs, 0 otherwise
    
    // Audio processing
    float* input_buffer;
    float* output_buffer;
    int frame_count;
    
    // RNNoise features
    float features[NB_FEATURES];
    float vad_prob;
    float gains[NB_BANDS];
    
    // RNNoise state for feature extraction
    DenoiseState* denoise_state;
    
    // Biquad filter memory for high-pass filtering
    float mem_hp_x[2];
    
    // Processing buffers (same as in rnnoise_process_frame)
    kiss_fft_cpx X[FREQ_SIZE];
    kiss_fft_cpx P[WINDOW_SIZE];
    float Ex[NB_BANDS], Ep[NB_BANDS];
    float Exp[NB_BANDS];
    float lastg[NB_BANDS];  // For gain smoothing
    
    // Synthesis memory for overlap-add
    float synthesis_mem[FRAME_SIZE];
    
    // GRU states for ONNX inference (external state management)
    float vad_gru_state[24];      // VAD GRU hidden state
    float noise_gru_state[48];    // Noise GRU hidden state  
    float denoise_gru_state[96];  // Denoise GRU hidden state
    int gru_states_initialized;   // Flag to track initialization
} RNNoiseContext;

// Function declarations
int load_wav_file(const char* filename, float** audio_data, int* num_samples, int* sample_rate);
int save_wav_file(const char* filename, const float* audio_data, int num_samples, int sample_rate);
int initialize_onnx_model(RNNoiseContext* ctx, const char* model_path);
int process_audio_frame(RNNoiseContext* ctx, const float* input_frame, float* output_frame);
void cleanup_context(RNNoiseContext* ctx);
int extract_features(RNNoiseContext* ctx, const float* frame, float* features);
void apply_gains(float* frame, const float* gains);

// GRU state management functions
void initialize_gru_states(RNNoiseContext* ctx);
int onnx_inference_with_states(RNNoiseContext* ctx, const float* features, float* gains, float* vad);

// Load WAV file
int load_wav_file(const char* filename, float** audio_data, int* num_samples, int* sample_rate) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        return -1;
    }
    
    WAVHeader header;
    if (fread(&header, sizeof(WAVHeader), 1, file) != 1) {
        fprintf(stderr, "Error: Cannot read WAV header\n");
        fclose(file);
        return -1;
    }
    
    // Validate WAV header
    if (memcmp(header.riff, "RIFF", 4) != 0 || 
        memcmp(header.wave, "WAVE", 4) != 0 ||
        memcmp(header.fmt, "fmt ", 4) != 0 ||
        memcmp(header.data, "data", 4) != 0) {
        fprintf(stderr, "Error: Invalid WAV file format\n");
        fclose(file);
        return -1;
    }
    
    if (header.audio_format != 1) {
        fprintf(stderr, "Error: Only PCM format is supported\n");
        fclose(file);
        return -1;
    }
    
    if (header.num_channels != 1) {
        fprintf(stderr, "Error: Only mono audio is supported\n");
        fclose(file);
        return -1;
    }
    
    if (header.bits_per_sample != 16) {
        fprintf(stderr, "Error: Only 16-bit audio is supported\n");
        fclose(file);
        return -1;
    }
    
    *sample_rate = header.sample_rate;
    *num_samples = header.data_size / sizeof(int16_t);
    
    // Allocate memory for audio data
    *audio_data = (float*)malloc(*num_samples * sizeof(float));
    if (!*audio_data) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(file);
        return -1;
    }
    
    // Read audio data
    int16_t* temp_buffer = (int16_t*)malloc(*num_samples * sizeof(int16_t));
    if (!temp_buffer) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        free(*audio_data);
        fclose(file);
        return -1;
    }
    
    if (fread(temp_buffer, sizeof(int16_t), *num_samples, file) != *num_samples) {
        fprintf(stderr, "Error: Cannot read audio data\n");
        free(*audio_data);
        free(temp_buffer);
        fclose(file);
        return -1;
    }
    
    // Convert to float (no range conversion, just type conversion)
    for (int i = 0; i < *num_samples; i++) {
        (*audio_data)[i] = (float)temp_buffer[i];
    }
    
    free(temp_buffer);
    fclose(file);
    
    printf("Loaded WAV file: %s\n", filename);
    printf("  Sample rate: %d Hz\n", *sample_rate);
    printf("  Duration: %.2f seconds\n", (float)*num_samples / *sample_rate);
    printf("  Samples: %d\n", *num_samples);
    
    return 0;
}

// Save WAV file
int save_wav_file(const char* filename, const float* audio_data, int num_samples, int sample_rate) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Error: Cannot create file %s\n", filename);
        return -1;
    }
    
    // Prepare WAV header
    WAVHeader header;
    memcpy(header.riff, "RIFF", 4);
    header.file_size = sizeof(WAVHeader) - 8 + num_samples * sizeof(int16_t);
    memcpy(header.wave, "WAVE", 4);
    memcpy(header.fmt, "fmt ", 4);
    header.fmt_size = 16;
    header.audio_format = 1;
    header.num_channels = 1;
    header.sample_rate = sample_rate;
    header.byte_rate = sample_rate * sizeof(int16_t);
    header.block_align = sizeof(int16_t);
    header.bits_per_sample = 16;
    memcpy(header.data, "data", 4);
    header.data_size = num_samples * sizeof(int16_t);
    
    // Write header
    if (fwrite(&header, sizeof(WAVHeader), 1, file) != 1) {
        fprintf(stderr, "Error: Cannot write WAV header\n");
        fclose(file);
        return -1;
    }
    
    // Convert and write audio data
    int16_t* temp_buffer = (int16_t*)malloc(num_samples * sizeof(int16_t));
    if (!temp_buffer) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(file);
        return -1;
    }
    
    for (int i = 0; i < num_samples; i++) {
        float sample = audio_data[i];
        // Clamp to 16-bit range [-32768, 32767]
        if (sample > 32767.0f) sample = 32767.0f;
        if (sample < -32768.0f) sample = -32768.0f;
        temp_buffer[i] = (int16_t)sample;
    }
    
    if (fwrite(temp_buffer, sizeof(int16_t), num_samples, file) != num_samples) {
        fprintf(stderr, "Error: Cannot write audio data\n");
        free(temp_buffer);
        fclose(file);
        return -1;
    }
    
    free(temp_buffer);
    fclose(file);
    
    printf("Saved WAV file: %s\n", filename);
    printf("  Sample rate: %d Hz\n", sample_rate);
    printf("  Duration: %.2f seconds\n", (float)num_samples / sample_rate);
    printf("  Samples: %d\n", num_samples);
    
    return 0;
}

// Initialize ONNX model
int initialize_onnx_model(RNNoiseContext* ctx, const char* model_path) {
    // Get ONNX Runtime API
    const OrtApiBase* api_base = OrtGetApiBase();
    if (!api_base) {
        fprintf(stderr, "Error getting ONNX Runtime API base\n");
        return -1;
    }
    
    ctx->api = api_base->GetApi(ORT_API_VERSION);
    if (!ctx->api) {
        fprintf(stderr, "Error getting ONNX Runtime API\n");
        return -1;
    }
    
    // Initialize ONNX Runtime environment
    OrtStatus* status = ctx->api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "RNNoiseONNX", &ctx->env);
    if (status != NULL) {
        fprintf(stderr, "Error creating ONNX Runtime environment\n");
        return -1;
    }
    
    // Create session options
    status = ctx->api->CreateSessionOptions(&ctx->session_options);
    if (status != NULL) {
        fprintf(stderr, "Error creating session options\n");
        return -1;
    }
    
    // Set session options
    status = ctx->api->SetIntraOpNumThreads(ctx->session_options, 1);
    if (status != NULL) {
        fprintf(stderr, "Error setting intra-op threads\n");
        return -1;
    }
    
    status = ctx->api->SetSessionGraphOptimizationLevel(ctx->session_options, ORT_ENABLE_EXTENDED);
    if (status != NULL) {
        fprintf(stderr, "Error setting optimization level\n");
        return -1;
    }
    
    // Create session
    status = ctx->api->CreateSession(ctx->env, model_path, ctx->session_options, &ctx->session);
    if (status != NULL) {
        fprintf(stderr, "Error creating ONNX session\n");
        return -1;
    }
    
    // Get allocator
    status = ctx->api->GetAllocatorWithDefaultOptions(&ctx->allocator);
    if (status != NULL) {
        fprintf(stderr, "Error getting allocator\n");
        return -1;
    }
    
    // Get input/output names
    size_t num_input_nodes, num_output_nodes;
    status = ctx->api->SessionGetInputCount(ctx->session, &num_input_nodes);
    if (status != NULL) {
        fprintf(stderr, "Error getting input count\n");
        return -1;
    }
    
    status = ctx->api->SessionGetOutputCount(ctx->session, &num_output_nodes);
    if (status != NULL) {
        fprintf(stderr, "Error getting output count\n");
        return -1;
    }
    
    printf("ONNX Model Info:\n");
    printf("  Input nodes: %zu\n", num_input_nodes);
    printf("  Output nodes: %zu\n", num_output_nodes);
    
    // Detect model type: 4 inputs + 5 outputs = model with GRU states
    ctx->has_gru_states = (num_input_nodes == 4 && num_output_nodes == 5);
    
    if (ctx->has_gru_states) {
        printf("  Model type: WITH GRU state inputs/outputs\n");
        
        // Get all input names
        status = ctx->api->SessionGetInputName(ctx->session, 0, ctx->allocator, &ctx->input_name);
        if (status != NULL) {
            fprintf(stderr, "Error getting features input name\n");
            return -1;
        }
        status = ctx->api->SessionGetInputName(ctx->session, 1, ctx->allocator, &ctx->input_name_vad_state);
        if (status != NULL) {
            fprintf(stderr, "Error getting VAD state input name\n");
            return -1;
        }
        status = ctx->api->SessionGetInputName(ctx->session, 2, ctx->allocator, &ctx->input_name_noise_state);
        if (status != NULL) {
            fprintf(stderr, "Error getting noise state input name\n");
            return -1;
        }
        status = ctx->api->SessionGetInputName(ctx->session, 3, ctx->allocator, &ctx->input_name_denoise_state);
        if (status != NULL) {
            fprintf(stderr, "Error getting denoise state input name\n");
            return -1;
        }
        
        // Get all output names
        status = ctx->api->SessionGetOutputName(ctx->session, 0, ctx->allocator, &ctx->output_name_denoise);
        if (status != NULL) {
            fprintf(stderr, "Error getting denoise output name\n");
            return -1;
        }
        status = ctx->api->SessionGetOutputName(ctx->session, 1, ctx->allocator, &ctx->output_name_vad);
        if (status != NULL) {
            fprintf(stderr, "Error getting VAD output name\n");
            return -1;
        }
        status = ctx->api->SessionGetOutputName(ctx->session, 2, ctx->allocator, &ctx->output_name_vad_state);
        if (status != NULL) {
            fprintf(stderr, "Error getting VAD state output name\n");
            return -1;
        }
        status = ctx->api->SessionGetOutputName(ctx->session, 3, ctx->allocator, &ctx->output_name_noise_state);
        if (status != NULL) {
            fprintf(stderr, "Error getting noise state output name\n");
            return -1;
        }
        status = ctx->api->SessionGetOutputName(ctx->session, 4, ctx->allocator, &ctx->output_name_denoise_state);
        if (status != NULL) {
            fprintf(stderr, "Error getting denoise state output name\n");
            return -1;
        }
        
        printf("  Inputs:\n");
        printf("    [0] %s (features)\n", ctx->input_name);
        printf("    [1] %s (VAD GRU state)\n", ctx->input_name_vad_state);
        printf("    [2] %s (noise GRU state)\n", ctx->input_name_noise_state);
        printf("    [3] %s (denoise GRU state)\n", ctx->input_name_denoise_state);
        printf("  Outputs:\n");
        printf("    [0] %s (denoise)\n", ctx->output_name_denoise);
        printf("    [1] %s (VAD)\n", ctx->output_name_vad);
        printf("    [2] %s (VAD GRU state)\n", ctx->output_name_vad_state);
        printf("    [3] %s (noise GRU state)\n", ctx->output_name_noise_state);
        printf("    [4] %s (denoise GRU state)\n", ctx->output_name_denoise_state);
    } else {
        printf("  Model type: Standard (without GRU state ports)\n");
        
        // Get input name (standard model)
        status = ctx->api->SessionGetInputName(ctx->session, 0, ctx->allocator, &ctx->input_name);
        if (status != NULL) {
            fprintf(stderr, "Error getting input name\n");
            return -1;
        }
        
        // Get output names (standard model)
        status = ctx->api->SessionGetOutputName(ctx->session, 0, ctx->allocator, &ctx->output_name_denoise);
        if (status != NULL) {
            fprintf(stderr, "Error getting denoise output name\n");
            return -1;
        }
        
        status = ctx->api->SessionGetOutputName(ctx->session, 1, ctx->allocator, &ctx->output_name_vad);
        if (status != NULL) {
            fprintf(stderr, "Error getting VAD output name\n");
            return -1;
        }
        
        printf("  Input: %s\n", ctx->input_name);
        printf("  Output denoise: %s\n", ctx->output_name_denoise);
        printf("  Output VAD: %s\n", ctx->output_name_vad);
    }
    
    // Allocate buffers
    ctx->input_buffer = (float*)malloc(FRAME_SIZE * sizeof(float));
    ctx->output_buffer = (float*)malloc(FRAME_SIZE * sizeof(float));
    
    if (!ctx->input_buffer || !ctx->output_buffer) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        return -1;
    }
    
    // Initialize RNNoise state for feature extraction
    ctx->denoise_state = rnnoise_create(NULL);
    if (!ctx->denoise_state) {
        fprintf(stderr, "Error: Failed to create RNNoise state\n");
        return -1;
    }
    rnnoise_init(ctx->denoise_state, NULL);
    
    // Initialize biquad filter memory
    ctx->mem_hp_x[0] = 0.0f;
    ctx->mem_hp_x[1] = 0.0f;
    
    // Initialize processing buffers
    memset(ctx->X, 0, sizeof(ctx->X));
    memset(ctx->P, 0, sizeof(ctx->P));
    memset(ctx->Ex, 0, sizeof(ctx->Ex));
    memset(ctx->Ep, 0, sizeof(ctx->Ep));
    memset(ctx->Exp, 0, sizeof(ctx->Exp));
    memset(ctx->lastg, 0, sizeof(ctx->lastg));
    memset(ctx->synthesis_mem, 0, sizeof(ctx->synthesis_mem));
    
    // Initialize frame count
    ctx->frame_count = 0;
    
    // Initialize GRU states if model supports it
    initialize_gru_states(ctx);
    
    printf("ONNX model loaded successfully: %s\n", model_path);
    return 0;
}

// Initialize GRU states
void initialize_gru_states(RNNoiseContext* ctx) {
    memset(ctx->vad_gru_state, 0, sizeof(ctx->vad_gru_state));
    memset(ctx->noise_gru_state, 0, sizeof(ctx->noise_gru_state));
    memset(ctx->denoise_gru_state, 0, sizeof(ctx->denoise_gru_state));
    ctx->gru_states_initialized = 0;
}

// Cross-platform timing function using C standard library
static double get_time_ms(void) {
    return (double)clock() * 1000.0 / CLOCKS_PER_SEC; // Convert to milliseconds
}

// ONNX inference with external state management
int onnx_inference_with_states(RNNoiseContext* ctx, const float* features, float* gains, float* vad) {
    // Prepare separate input tensors for features and GRU states
    float features_data[42];
    float vad_state_data[24];
    float noise_state_data[48];
    float denoise_state_data[96];
    
    // Copy features
    memcpy(features_data, features, 42 * sizeof(float));
    
    // Copy GRU states (use saved states for next frame)
    memcpy(vad_state_data, ctx->vad_gru_state, 24 * sizeof(float));
    memcpy(noise_state_data, ctx->noise_gru_state, 48 * sizeof(float));
    memcpy(denoise_state_data, ctx->denoise_gru_state, 96 * sizeof(float));
    
    // Create input tensors
    const int64_t features_shape[] = {1, 1, 42};
    const int64_t vad_state_shape[] = {1, 24};
    const int64_t noise_state_shape[] = {1, 48};
    const int64_t denoise_state_shape[] = {1, 96};
    
    OrtMemoryInfo* memory_info;
    OrtStatus* status = ctx->api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
    if (status != NULL) {
        fprintf(stderr, "Error creating memory info\n");
        return -1;
    }
    
    // Create input tensors
    OrtValue* features_tensor = NULL;
    OrtValue* vad_state_tensor = NULL;
    OrtValue* noise_state_tensor = NULL;
    OrtValue* denoise_state_tensor = NULL;
    
    status = ctx->api->CreateTensorWithDataAsOrtValue(
        memory_info, features_data, 42 * sizeof(float),
        features_shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &features_tensor);
    if (status != NULL) {
        fprintf(stderr, "Error creating features tensor\n");
        ctx->api->ReleaseMemoryInfo(memory_info);
        return -1;
    }
    
    status = ctx->api->CreateTensorWithDataAsOrtValue(
        memory_info, vad_state_data, 24 * sizeof(float),
        vad_state_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &vad_state_tensor);
    if (status != NULL) {
        fprintf(stderr, "Error creating VAD state tensor\n");
        ctx->api->ReleaseValue(features_tensor);
        ctx->api->ReleaseMemoryInfo(memory_info);
        return -1;
    }
    
    status = ctx->api->CreateTensorWithDataAsOrtValue(
        memory_info, noise_state_data, 48 * sizeof(float),
        noise_state_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &noise_state_tensor);
    if (status != NULL) {
        fprintf(stderr, "Error creating noise state tensor\n");
        ctx->api->ReleaseValue(features_tensor);
        ctx->api->ReleaseValue(vad_state_tensor);
        ctx->api->ReleaseMemoryInfo(memory_info);
        return -1;
    }
    
    status = ctx->api->CreateTensorWithDataAsOrtValue(
        memory_info, denoise_state_data, 96 * sizeof(float),
        denoise_state_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &denoise_state_tensor);
    if (status != NULL) {
        fprintf(stderr, "Error creating denoise state tensor\n");
        ctx->api->ReleaseValue(features_tensor);
        ctx->api->ReleaseValue(vad_state_tensor);
        ctx->api->ReleaseValue(noise_state_tensor);
        ctx->api->ReleaseMemoryInfo(memory_info);
        return -1;
    }
    
    // Prepare input names and tensors
    const char* input_names[] = {ctx->input_name, ctx->input_name_vad_state, 
                                 ctx->input_name_noise_state, ctx->input_name_denoise_state};
    OrtValue* input_tensors[] = {features_tensor, vad_state_tensor, noise_state_tensor, denoise_state_tensor};
    
    // Prepare output names
    const char* output_names[] = {ctx->output_name_denoise, ctx->output_name_vad, 
                                  ctx->output_name_vad_state, ctx->output_name_noise_state, 
                                  ctx->output_name_denoise_state};
    OrtValue* output_tensors[5] = {NULL, NULL, NULL, NULL, NULL};
    
    // Run inference
    status = ctx->api->Run(ctx->session, NULL, input_names, (const OrtValue* const*)input_tensors, 4,
                   output_names, 5, output_tensors);
    if (status != NULL) {
        fprintf(stderr, "Error running inference\n");
        ctx->api->ReleaseValue(features_tensor);
        ctx->api->ReleaseValue(vad_state_tensor);
        ctx->api->ReleaseValue(noise_state_tensor);
        ctx->api->ReleaseValue(denoise_state_tensor);
        ctx->api->ReleaseMemoryInfo(memory_info);
        return -1;
    }
    
    // Get output data
    float* denoise_output = NULL;
    float* vad_output = NULL;
    float* updated_vad_state = NULL;
    float* updated_noise_state = NULL;
    float* updated_denoise_state = NULL;
    
    status = ctx->api->GetTensorMutableData(output_tensors[0], (void**)&denoise_output);
    if (status != NULL) {
        fprintf(stderr, "Error getting denoise output data\n");
        goto cleanup;
    }
    
    status = ctx->api->GetTensorMutableData(output_tensors[1], (void**)&vad_output);
    if (status != NULL) {
        fprintf(stderr, "Error getting VAD output data\n");
        goto cleanup;
    }
    
    status = ctx->api->GetTensorMutableData(output_tensors[2], (void**)&updated_vad_state);
    if (status != NULL) {
        fprintf(stderr, "Error getting updated VAD state data\n");
        goto cleanup;
    }
    
    status = ctx->api->GetTensorMutableData(output_tensors[3], (void**)&updated_noise_state);
    if (status != NULL) {
        fprintf(stderr, "Error getting updated noise state data\n");
        goto cleanup;
    }
    
    status = ctx->api->GetTensorMutableData(output_tensors[4], (void**)&updated_denoise_state);
    if (status != NULL) {
        fprintf(stderr, "Error getting updated denoise state data\n");
        goto cleanup;
    }
    
    // Store results
    memcpy(gains, denoise_output, NB_BANDS * sizeof(float));
    *vad = vad_output[0];
    
    // Update GRU states with the outputs from the model (for next frame)
    memcpy(ctx->vad_gru_state, updated_vad_state, 24 * sizeof(float));
    memcpy(ctx->noise_gru_state, updated_noise_state, 48 * sizeof(float));
    memcpy(ctx->denoise_gru_state, updated_denoise_state, 96 * sizeof(float));
    ctx->gru_states_initialized = 1;
    
cleanup:
    // Cleanup
    ctx->api->ReleaseValue(features_tensor);
    ctx->api->ReleaseValue(vad_state_tensor);
    ctx->api->ReleaseValue(noise_state_tensor);
    ctx->api->ReleaseValue(denoise_state_tensor);
    for (int i = 0; i < 5; i++) {
        if (output_tensors[i]) {
            ctx->api->ReleaseValue(output_tensors[i]);
        }
    }
    ctx->api->ReleaseMemoryInfo(memory_info);
    
    return 0;
}

// Biquad filter function (same as in denoise.c)
static void biquad(float *y, float mem[2], const float *x, const float *b, const float *a, int N) {
    int i;
    for (i=0;i<N;i++) {
        float xi, yi;
        xi = x[i];
        yi = x[i] + mem[0];
        mem[0] = mem[1] + (b[0]*(double)xi - a[0]*(double)yi);
        mem[1] = (b[1]*(double)xi - a[1]*(double)yi);
        y[i] = yi;
    }
}

static void compute_band_energy(float *bandE, const kiss_fft_cpx *X) {
    int i;
    float sum[NB_BANDS] = {0};
    for (i=0;i<NB_BANDS-1;i++)
    {
      int j;
      int band_size;
      band_size = (eband5ms[i+1]-eband5ms[i])<<FRAME_SIZE_SHIFT;
      for (j=0;j<band_size;j++) {
        float tmp;
        float frac = (float)j/band_size;
        tmp = SQUARE(X[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j].r);
        tmp += SQUARE(X[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j].i);
        sum[i] += (1-frac)*tmp;
        sum[i+1] += frac*tmp;
      }
    }
    sum[0] *= 2;
    sum[NB_BANDS-1] *= 2;
    for (i=0;i<NB_BANDS;i++)
    {
      bandE[i] = sum[i];
    }
}

// Interpolation function (same as in denoise.c)
static void interp_band_gain(float *g, const float *bandE) {
    int i;
    memset(g, 0, FREQ_SIZE);
    for (i=0;i<NB_BANDS-1;i++) {
        int j;
        int band_size;
        band_size = (eband5ms[i+1]-eband5ms[i])<<FRAME_SIZE_SHIFT;
        for (j=0;j<band_size;j++) {
            float frac = (float)j/band_size;
            g[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j] = (1-frac)*bandE[i] + frac*bandE[i+1];
        }
    }
}

// Pitch filter function (same as in denoise.c)
static void pitch_filter(kiss_fft_cpx *X, const kiss_fft_cpx *P, const float *Ex, const float *Ep,
                  const float *Exp, const float *g) {
    int i;
    float r[NB_BANDS];
    float rf[FREQ_SIZE] = {0};
    for (i=0;i<NB_BANDS;i++) {
        if (Exp[i]>g[i]) r[i] = 1;
        else r[i] = SQUARE(Exp[i])*(1-SQUARE(g[i]))/(.001 + SQUARE(g[i])*(1-SQUARE(Exp[i])));
        r[i] = sqrt(MIN16(1, MAX16(0, r[i])));
        r[i] *= sqrt(Ex[i]/(1e-8+Ep[i]));
    }
    interp_band_gain(rf, r);
    for (i=0;i<FREQ_SIZE;i++) {
        X[i].r += rf[i]*P[i].r;
        X[i].i += rf[i]*P[i].i;
    }
    float newE[NB_BANDS];
    compute_band_energy(newE, X);
    float norm[NB_BANDS];
    float normf[FREQ_SIZE]={0};
    for (i=0;i<NB_BANDS;i++) {
        norm[i] = sqrt(Ex[i]/(1e-8+newE[i]));
    }
    interp_band_gain(normf, norm);
    for (i=0;i<FREQ_SIZE;i++) {
        X[i].r *= normf[i];
        X[i].i *= normf[i];
    }
}

// Inverse transform function (same as in denoise.c)
static void inverse_transform(float *out, const kiss_fft_cpx *in) {
    int i;
    kiss_fft_cpx x[WINDOW_SIZE];
    kiss_fft_cpx y[WINDOW_SIZE];
    for (i=0;i<FREQ_SIZE;i++) {
        x[i] = in[i];
    }
    for (;i<WINDOW_SIZE;i++) {
        x[i].r = x[WINDOW_SIZE - i].r;
        x[i].i = -x[WINDOW_SIZE - i].i;
    }
    // Use proper FFT - create a temporary FFT state
    kiss_fft_state *kfft = rnn_fft_alloc(WINDOW_SIZE, NULL, NULL, 0);
    if (kfft) {
        rnn_fft_c(kfft, x, y);
        rnn_fft_free(kfft, 0);
    } else {
        // Fallback: copy data without FFT
        for (i=0;i<WINDOW_SIZE;i++) {
            y[i].r = x[i].r;
            y[i].i = x[i].i;
        }
    }
    /* output in reverse order for IFFT. */
    out[0] = WINDOW_SIZE*y[0].r;
    for (i=1;i<WINDOW_SIZE;i++) {
        out[i] = WINDOW_SIZE*y[WINDOW_SIZE - i].r;
    }
}

// Apply window function (same as in denoise.c)
static void apply_window(float *x) {
    int i;
    // Use proper window coefficients as in denoise.c
    for (i=0;i<FRAME_SIZE;i++) {
        float window_coeff = sin(.5*M_PI*sin(.5*M_PI*(i+.5)/FRAME_SIZE) * sin(.5*M_PI*(i+.5)/FRAME_SIZE));
        x[i] *= window_coeff;
        x[WINDOW_SIZE - 1 - i] *= window_coeff;
    }
}

// Frame synthesis function (same as in denoise.c)
static void frame_synthesis(RNNoiseContext *ctx, float *out, const kiss_fft_cpx *y) {
    float x[WINDOW_SIZE];
    int i;
    inverse_transform(x, y);
    apply_window(x);
    // Overlap-add with synthesis memory
    for (i=0;i<FRAME_SIZE;i++) out[i] = x[i] + ctx->synthesis_mem[i];
    // Update synthesis memory for next frame
    RNN_COPY(ctx->synthesis_mem, &x[FRAME_SIZE], FRAME_SIZE);
}


// Apply gains to audio frame
void apply_gains(float* frame, const float* gains) {
    for (int i = 0; i < FRAME_SIZE; i++) {
        int band = (i * NB_BANDS) / FRAME_SIZE;
        if (band >= NB_BANDS) band = NB_BANDS - 1;
        frame[i] *= gains[band];
    }
}

// Process single audio frame (following rnnoise_process_frame flow)
int process_audio_frame(RNNoiseContext* ctx, const float* input_frame, float* output_frame) {
    int i;
    float x[FRAME_SIZE];
    float g[NB_BANDS];
    float g_c_version[NB_BANDS];
    float gf[FREQ_SIZE] = {1};
    float vad_prob = 0;
    float vad_prob_c_version = 0;
    int silence;
    
    // Increment frame count
    ctx->frame_count++;
    
    // Apply biquad high-pass filter (same as in rnnoise_process_frame)
    static const float a_hp[2] = {-1.99599, 0.99600};
    static const float b_hp[2] = {-2, 1};
    biquad(x, ctx->mem_hp_x, input_frame, b_hp, a_hp, FRAME_SIZE);
    
    // Extract features using RNNoise's compute_frame_features
    silence = compute_frame_features(ctx->denoise_state, ctx->X, ctx->P, ctx->Ex, ctx->Ep, ctx->Exp, ctx->features, x);
    
    if (!silence) {
        // Start timing for ONNX inference
        double inference_start_time = get_time_ms();
        
        // Use appropriate inference method based on model type
        if (ctx->has_gru_states) {
            // Use ONNX inference with GRU state management (streaming inference)
            if (onnx_inference_with_states(ctx, ctx->features, g, &vad_prob) != 0) {
                fprintf(stderr, "Error in ONNX inference with states\n");
                return -1;
            }
            
            // Debug output for first few frames
            if (ctx->frame_count < 5) {
                printf("Frame %d - ONNX with state management:\n", ctx->frame_count);
                printf("  VAD GRU state[0-3]: %.6f %.6f %.6f %.6f\n", 
                       ctx->vad_gru_state[0], ctx->vad_gru_state[1], 
                       ctx->vad_gru_state[2], ctx->vad_gru_state[3]);
                printf("  Noise GRU state[0-3]: %.6f %.6f %.6f %.6f\n", 
                       ctx->noise_gru_state[0], ctx->noise_gru_state[1], 
                       ctx->noise_gru_state[2], ctx->noise_gru_state[3]);
                printf("  Denoise GRU state[0-3]: %.6f %.6f %.6f %.6f\n", 
                       ctx->denoise_gru_state[0], ctx->denoise_gru_state[1], 
                       ctx->denoise_gru_state[2], ctx->denoise_gru_state[3]);
                printf("  VAD: %.6f\n", vad_prob);
                for (i = 0; i < 5; i++) {
                    printf("  Gain[%d]: %.6f\n", i, g[i]);
                }
            }
        } else {
            // Use standard ONNX inference (without state management)
            float input_tensor_values[NB_FEATURES];
            memcpy(input_tensor_values, ctx->features, NB_FEATURES * sizeof(float));
            
            // Create input tensor
            const int64_t input_shape[] = {1, 1, NB_FEATURES};
            const size_t input_tensor_size = NB_FEATURES * sizeof(float);
            
            OrtMemoryInfo* memory_info;
            OrtStatus* status = ctx->api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
            if (status != NULL) {
                fprintf(stderr, "Error creating memory info\n");
                return -1;
            }
            
            OrtValue* input_tensor = NULL;
            status = ctx->api->CreateTensorWithDataAsOrtValue(
                memory_info, input_tensor_values, input_tensor_size,
                input_shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor);
            if (status != NULL) {
                fprintf(stderr, "Error creating input tensor\n");
                ctx->api->ReleaseMemoryInfo(memory_info);
                return -1;
            }
            
            // Prepare output tensors
            const char* input_names[] = {ctx->input_name};
            const char* output_names[] = {ctx->output_name_denoise, ctx->output_name_vad};
            OrtValue* output_tensors[2] = {NULL, NULL};
            
            // Run inference
            status = ctx->api->Run(ctx->session, NULL, input_names, (const OrtValue* const*)&input_tensor, 1,
                           output_names, 2, output_tensors);
            if (status != NULL) {
                fprintf(stderr, "Error running inference\n");
                ctx->api->ReleaseValue(input_tensor);
                ctx->api->ReleaseMemoryInfo(memory_info);
                return -1;
            }
            
            // Get output data
            float* denoise_output = NULL;
            float* vad_output = NULL;
            
            status = ctx->api->GetTensorMutableData(output_tensors[0], (void**)&denoise_output);
            if (status != NULL) {
                fprintf(stderr, "Error getting denoise output data\n");
                ctx->api->ReleaseValue(input_tensor);
                ctx->api->ReleaseValue(output_tensors[0]);
                ctx->api->ReleaseValue(output_tensors[1]);
                ctx->api->ReleaseMemoryInfo(memory_info);
                return -1;
            }
            
            status = ctx->api->GetTensorMutableData(output_tensors[1], (void**)&vad_output);
            if (status != NULL) {
                fprintf(stderr, "Error getting VAD output data\n");
                ctx->api->ReleaseValue(input_tensor);
                ctx->api->ReleaseValue(output_tensors[0]);
                ctx->api->ReleaseValue(output_tensors[1]);
                ctx->api->ReleaseMemoryInfo(memory_info);
                return -1;
            }
            
            // Store results
            memcpy(g, denoise_output, NB_BANDS * sizeof(float));
            vad_prob = vad_output[0];
            
            // Cleanup ONNX tensors
            ctx->api->ReleaseValue(input_tensor);
            ctx->api->ReleaseValue(output_tensors[0]);
            ctx->api->ReleaseValue(output_tensors[1]);
            ctx->api->ReleaseMemoryInfo(memory_info);
        }
        
        // End timing and print statistics
        double inference_end_time = get_time_ms();
        double inference_time = inference_end_time - inference_start_time;
        
        // Start timing for C inference
        double c_inference_start_time = get_time_ms();
        
        {
            compute_rnn_c(ctx->denoise_state, g_c_version, &vad_prob_c_version, ctx->features);
        }
        
        // End timing for C inference
        double c_inference_end_time = get_time_ms();
        double c_inference_time = c_inference_end_time - c_inference_start_time;
        
        // Print timing statistics for first few frames
        if (ctx->frame_count <= 15) {
            printf("Frame %d - ONNX inference time: %.3f ms, C inference time: %.3f ms (ratio: %.2fx)\n", 
                   ctx->frame_count, inference_time, c_inference_time, 
                   c_inference_time > 0 ? inference_time / c_inference_time : 0.0);
        }
        
        // Apply pitch filter (same as in rnnoise_process_frame)
        pitch_filter(ctx->X, ctx->P, ctx->Ex, ctx->Ep, ctx->Exp, g);
        
        // Apply gain smoothing (same as in rnnoise_process_frame)
        for (i=0;i<NB_BANDS;i++) {
            float alpha = .6f;
            g[i] = MAX16(g[i], alpha*ctx->lastg[i]);
            ctx->lastg[i] = g[i];
        }
        
        // Interpolate band gains to frequency domain
        interp_band_gain(gf, g);
        
        // Apply gains to frequency domain (same as in rnnoise_process_frame)
        for (i=0;i<FREQ_SIZE;i++) {
            ctx->X[i].r *= gf[i];
            ctx->X[i].i *= gf[i];
        }
    }
    
    // Frame synthesis (same as in rnnoise_process_frame)
    frame_synthesis(ctx, output_frame, ctx->X);
    
    return vad_prob;
}

// Cleanup context
void cleanup_context(RNNoiseContext* ctx) {
    if (ctx->denoise_state) {
        rnnoise_destroy(ctx->denoise_state);
        ctx->denoise_state = NULL;
    }
    if (ctx->input_buffer) {
        free(ctx->input_buffer);
        ctx->input_buffer = NULL;
    }
    if (ctx->output_buffer) {
        free(ctx->output_buffer);
        ctx->output_buffer = NULL;
    }
    if (ctx->api) {
        if (ctx->input_name) {
            ctx->api->AllocatorFree(ctx->allocator, ctx->input_name);
            ctx->input_name = NULL;
        }
        if (ctx->input_name_vad_state) {
            ctx->api->AllocatorFree(ctx->allocator, ctx->input_name_vad_state);
            ctx->input_name_vad_state = NULL;
        }
        if (ctx->input_name_noise_state) {
            ctx->api->AllocatorFree(ctx->allocator, ctx->input_name_noise_state);
            ctx->input_name_noise_state = NULL;
        }
        if (ctx->input_name_denoise_state) {
            ctx->api->AllocatorFree(ctx->allocator, ctx->input_name_denoise_state);
            ctx->input_name_denoise_state = NULL;
        }
        if (ctx->output_name_denoise) {
            ctx->api->AllocatorFree(ctx->allocator, ctx->output_name_denoise);
            ctx->output_name_denoise = NULL;
        }
        if (ctx->output_name_vad) {
            ctx->api->AllocatorFree(ctx->allocator, ctx->output_name_vad);
            ctx->output_name_vad = NULL;
        }
        if (ctx->output_name_vad_state) {
            ctx->api->AllocatorFree(ctx->allocator, ctx->output_name_vad_state);
            ctx->output_name_vad_state = NULL;
        }
        if (ctx->output_name_noise_state) {
            ctx->api->AllocatorFree(ctx->allocator, ctx->output_name_noise_state);
            ctx->output_name_noise_state = NULL;
        }
        if (ctx->output_name_denoise_state) {
            ctx->api->AllocatorFree(ctx->allocator, ctx->output_name_denoise_state);
            ctx->output_name_denoise_state = NULL;
        }
        if (ctx->session) {
            ctx->api->ReleaseSession(ctx->session);
            ctx->session = NULL;
        }
        if (ctx->session_options) {
            ctx->api->ReleaseSessionOptions(ctx->session_options);
            ctx->session_options = NULL;
        }
        if (ctx->env) {
            ctx->api->ReleaseEnv(ctx->env);
            ctx->env = NULL;
        }
    }
}

// Main function
int main(int argc, char* argv[]) {
    if (argc != 4) {
        printf("Usage: %s <input.wav> <output.wav> <model.onnx>\n", argv[0]);
        printf("  input.wav  - Input audio file (16-bit mono WAV)\n");
        printf("  output.wav - Output audio file (16-bit mono WAV)\n");
        printf("  model.onnx - ONNX model file\n");
        return 1;
    }
    
    const char* input_file = argv[1];
    const char* output_file = argv[2];
    const char* model_file = argv[3];
    
    printf("RNNoise ONNX Audio Denoiser\n");
    printf("============================\n");
    printf("Input file: %s\n", input_file);
    printf("Output file: %s\n", output_file);
    printf("Model file: %s\n", model_file);
    printf("\n");
    
    // Load input audio
    float* audio_data;
    int num_samples, sample_rate;
    
    if (load_wav_file(input_file, &audio_data, &num_samples, &sample_rate) != 0) {
        return 1;
    }
    
    if (sample_rate != SAMPLE_RATE) {
        fprintf(stderr, "Error: Sample rate must be %d Hz, got %d Hz\n", SAMPLE_RATE, sample_rate);
        free(audio_data);
        return 1;
    }
    
    // Initialize ONNX model
    RNNoiseContext ctx = {0};
    if (initialize_onnx_model(&ctx, model_file) != 0) {
        free(audio_data);
        return 1;
    }
    
    // Process audio
    int num_frames = (num_samples + FRAME_SIZE - 1) / FRAME_SIZE;
    float* output_audio = (float*)malloc(num_samples * sizeof(float));
    if (!output_audio) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        free(audio_data);
        cleanup_context(&ctx);
        return 1;
    }
    
    printf("\nProcessing audio...\n");
    printf("  Frames: %d\n", num_frames);
    printf("  Frame size: %d samples\n", FRAME_SIZE);
    printf("  Frame duration: %.2f ms\n", (float)FRAME_SIZE / SAMPLE_RATE * 1000.0f);
    
    int processed_frames = 0;
    int speech_frames = 0;
    
    for (int frame_idx = 0; frame_idx < num_frames; frame_idx++) {
        // Prepare input frame
        int start_sample = frame_idx * FRAME_SIZE;
        int end_sample = start_sample + FRAME_SIZE;
        if (end_sample > num_samples) end_sample = num_samples;
        
        // Pad with zeros if necessary
        memset(ctx.input_buffer, 0, FRAME_SIZE * sizeof(float));
        int frame_samples = end_sample - start_sample;
        memcpy(ctx.input_buffer, &audio_data[start_sample], frame_samples * sizeof(float));
        
        // Process frame
        if (process_audio_frame(&ctx, ctx.input_buffer, ctx.output_buffer) != 0) {
            fprintf(stderr, "Error processing frame %d\n", frame_idx);
            free(audio_data);
            free(output_audio);
            cleanup_context(&ctx);
            return 1;
        }
        
        // Copy output frame
        memcpy(&output_audio[start_sample], ctx.output_buffer, frame_samples * sizeof(float));
        
        // Statistics
        processed_frames++;
        if (ctx.vad_prob > 0.5f) {
            speech_frames++;
        }
        
        // Progress indicator
        if (frame_idx % 100 == 0) {
            printf("  Processed %d/%d frames (%.1f%%)\n", 
                frame_idx, num_frames, (float)frame_idx / num_frames * 100.0f);
        }
    }
    
    printf("  Processed %d/%d frames (100.0%%)\n", processed_frames, num_frames);
    printf("  Speech frames: %d (%.1f%%)\n", speech_frames, (float)speech_frames / processed_frames * 100.0f);
    
    // Save output audio
    if (save_wav_file(output_file, output_audio, num_samples, sample_rate) != 0) {
        free(audio_data);
        free(output_audio);
        cleanup_context(&ctx);
        return 1;
    }
    
    // Cleanup
    free(audio_data);
    free(output_audio);
    cleanup_context(&ctx);
    
    printf("\nProcessing completed successfully!\n");
    return 0;
}
