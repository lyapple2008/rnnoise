/* Copyright (c) 2018 Gregor Richards
 * Copyright (c) 2017 Mozilla */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "rnnoise.h"
#include <samplerate.h>

#ifdef _WIN32
#define strcasecmp _stricmp
#endif

#define FRAME_SIZE 480
#define TARGET_SAMPLE_RATE 48000
#define FRAME_TIME_MS 10  // 10ms frame time

// Resampling context structure
typedef struct {
    SRC_STATE *src_state;
    int input_rate;
    int output_rate;
    int channels;
    float *input_buffer;
    float *output_buffer;
    int input_buffer_size;
    int output_buffer_size;
    int input_buffer_filled;
    int output_buffer_filled;
    int output_buffer_pos;
} ResampleContext;

// Function declarations
ResampleContext* init_resample_context(int input_rate, int output_rate, int channels);
void free_resample_context(ResampleContext *ctx);
int resample_audio(ResampleContext *ctx, float *input, int input_frames, 
                   float *output, int *output_frames);
int calculate_frame_size(int sample_rate);

// WAV file header structure
typedef struct {
    char riff[4];           // "RIFF"
    uint32_t file_size;     // File size - 8
    char wave[4];           // "WAVE"
    char fmt[4];            // "fmt "
    uint32_t fmt_size;      // Format chunk size (16 for PCM)
    uint16_t audio_format;  // Audio format (1 for PCM)
    uint16_t num_channels;  // Number of channels
    uint32_t sample_rate;   // Sample rate
    uint32_t byte_rate;     // Byte rate
    uint16_t block_align;   // Block align
    uint16_t bits_per_sample; // Bits per sample
    char data[4];           // "data"
    uint32_t data_size;     // Data size
} WavHeader;

// Function to read WAV header
int read_wav_header(FILE *file, WavHeader *header, int *samples_count) {
    if (fread(header, sizeof(WavHeader), 1, file) != 1) {
        fprintf(stderr, "Error reading WAV header\n");
        return 0;
    }
    
    // Check if it's a valid WAV file
    if (strncmp(header->riff, "RIFF", 4) != 0 || 
        strncmp(header->wave, "WAVE", 4) != 0 ||
        strncmp(header->data, "data", 4) != 0) {
        fprintf(stderr, "Not a valid WAV file\n");
        return 0;
    }
    
    // Check if it's PCM format
    if (header->audio_format != 1) {
        fprintf(stderr, "Only PCM format is supported\n");
        return 0;
    }
    
    // Check if it's 16-bit
    if (header->bits_per_sample != 16) {
        fprintf(stderr, "Only 16-bit samples are supported\n");
        return 0;
    }
    
    // Calculate number of samples
    *samples_count = header->data_size / (header->num_channels * (header->bits_per_sample / 8));
    
    printf("WAV file info:\n");
    printf("  Channels: %d\n", header->num_channels);
    printf("  Sample rate: %d Hz\n", header->sample_rate);
    printf("  Bits per sample: %d\n", header->bits_per_sample);
    printf("  Total samples: %d\n", *samples_count);
    
    return 1;
}

// Function to write WAV header
int write_wav_header(FILE *file, int sample_rate, int num_channels, int samples_count) {
    WavHeader header;
    
    // Fill header
    strncpy(header.riff, "RIFF", 4);
    strncpy(header.wave, "WAVE", 4);
    strncpy(header.fmt, "fmt ", 4);
    strncpy(header.data, "data", 4);
    
    header.fmt_size = 16;
    header.audio_format = 1; // PCM
    header.num_channels = num_channels;
    header.sample_rate = sample_rate;
    header.bits_per_sample = 16;
    header.block_align = num_channels * (header.bits_per_sample / 8);
    header.byte_rate = sample_rate * header.block_align;
    header.data_size = samples_count * header.block_align;
    header.file_size = header.data_size + sizeof(WavHeader) - 8;
    
    if (fwrite(&header, sizeof(WavHeader), 1, file) != 1) {
        fprintf(stderr, "Error writing WAV header\n");
        return 0;
    }
    
    return 1;
}

// Function to check if file is WAV format
int is_wav_file(const char *filename) {
    const char *ext = strrchr(filename, '.');
    return ext && (strcasecmp(ext, ".wav") == 0);
}

// Function to update WAV header with final file size
void update_wav_header_size(FILE *file, int total_samples, int num_channels) {
    long current_pos = ftell(file);
    fseek(file, 4, SEEK_SET);
    uint32_t file_size = current_pos - 8;
    fwrite(&file_size, sizeof(uint32_t), 1, file);
    
    fseek(file, 40, SEEK_SET);
    uint32_t data_size = total_samples * num_channels * 2; // 16-bit samples
    fwrite(&data_size, sizeof(uint32_t), 1, file);
    
    fseek(file, current_pos, SEEK_SET);
}

// Initialize resampling context
ResampleContext* init_resample_context(int input_rate, int output_rate, int channels) {
    ResampleContext *ctx = malloc(sizeof(ResampleContext));
    if (!ctx) return NULL;
    
    ctx->input_rate = input_rate;
    ctx->output_rate = output_rate;
    ctx->channels = channels;
    ctx->input_buffer_filled = 0;
    ctx->output_buffer_filled = 0;
    ctx->output_buffer_pos = 0;
    
    // Calculate buffer sizes (with some extra space for resampling)
    ctx->input_buffer_size = FRAME_SIZE * 4;  // Extra space for resampling
    ctx->output_buffer_size = FRAME_SIZE * 4;
    
    ctx->input_buffer = malloc(ctx->input_buffer_size * sizeof(float));
    ctx->output_buffer = malloc(ctx->output_buffer_size * sizeof(float));
    
    if (!ctx->input_buffer || !ctx->output_buffer) {
        free_resample_context(ctx);
        return NULL;
    }
    
    // Initialize libsamplerate
    int error;
    ctx->src_state = src_new(SRC_SINC_BEST_QUALITY, channels, &error);
    if (!ctx->src_state) {
        fprintf(stderr, "Error initializing resampler: %s\n", src_strerror(error));
        free_resample_context(ctx);
        return NULL;
    }
    
    return ctx;
}

// Free resampling context
void free_resample_context(ResampleContext *ctx) {
    if (!ctx) return;
    
    if (ctx->src_state) {
        src_delete(ctx->src_state);
    }
    if (ctx->input_buffer) {
        free(ctx->input_buffer);
    }
    if (ctx->output_buffer) {
        free(ctx->output_buffer);
    }
    free(ctx);
}

// Resample audio data
int resample_audio(ResampleContext *ctx, float *input, int input_frames, 
                   float *output, int *output_frames) {
    SRC_DATA src_data;
    int error;
    
    src_data.data_in = input;
    src_data.data_out = output;
    src_data.input_frames = input_frames;
    src_data.output_frames = *output_frames;
    src_data.src_ratio = (double)ctx->output_rate / ctx->input_rate;
    src_data.end_of_input = 0;
    
    error = src_process(ctx->src_state, &src_data);
    if (error) {
        fprintf(stderr, "Resampling error: %s\n", src_strerror(error));
        return 0;
    }
    
    *output_frames = src_data.output_frames_gen;
    return 1;
}

// Calculate frame size for 10ms at given sample rate
int calculate_frame_size(int sample_rate) {
    return (sample_rate * FRAME_TIME_MS) / 1000;
}

int main(int argc, char **argv) {
  int i;
  int first = 1;
  float x[FRAME_SIZE];
  float y[FRAME_SIZE];  // Output frame for resampling
  FILE *f1, *fout;
  DenoiseState *st;
  WavHeader input_header;
  int input_samples_count = 0;
  int output_samples_count = 0;
  ResampleContext *input_resampler = NULL;
  ResampleContext *output_resampler = NULL;
  int input_sample_rate;
  int output_sample_rate;
  int input_frame_size;
  int output_frame_size;
  
  st = rnnoise_create(NULL);
  if (argc!=3) {
    fprintf(stderr, "usage: %s <noisy speech> <output denoised>\n", argv[0]);
    fprintf(stderr, "Only WAV format is supported\n");
    return 1;
  }
  
  // Check file formats - only WAV is supported
  if (!is_wav_file(argv[1])) {
    fprintf(stderr, "Error: Input file must be WAV format (.wav extension)\n");
    rnnoise_destroy(st);
    return 1;
  }
  
  if (!is_wav_file(argv[2])) {
    fprintf(stderr, "Error: Output file must be WAV format (.wav extension)\n");
    rnnoise_destroy(st);
    return 1;
  }
  
  f1 = fopen(argv[1], "rb");
  if (!f1) {
    fprintf(stderr, "Error opening input file: %s\n", argv[1]);
    rnnoise_destroy(st);
    return 1;
  }
  
  fout = fopen(argv[2], "wb");
  if (!fout) {
    fprintf(stderr, "Error opening output file: %s\n", argv[2]);
    fclose(f1);
    rnnoise_destroy(st);
    return 1;
  }
  
  // Read WAV header
  if (!read_wav_header(f1, &input_header, &input_samples_count)) {
    fprintf(stderr, "Error reading WAV header from input file\n");
    fclose(f1);
    fclose(fout);
    rnnoise_destroy(st);
    return 1;
  }
  
  input_sample_rate = input_header.sample_rate;
  output_sample_rate = input_sample_rate;  // Output will have same rate as input
  
  // Calculate frame sizes for 10ms
  input_frame_size = calculate_frame_size(input_sample_rate);
  output_frame_size = calculate_frame_size(output_sample_rate);
  
  printf("Input frame size: %d samples (10ms at %d Hz)\n", input_frame_size, input_sample_rate);
  printf("Output frame size: %d samples (10ms at %d Hz)\n", output_frame_size, output_sample_rate);
  
  // Check if input is mono (RNNoise works on mono audio)
  if (input_header.num_channels != 1) {
    fprintf(stderr, "Warning: Input has %d channels, RNNoise works on mono audio.\n", input_header.num_channels);
    fprintf(stderr, "Only the first channel will be processed.\n");
    fclose(f1);
    fclose(fout);
    rnnoise_destroy(st);
    return 1;
  }
  
  // Initialize resamplers if needed
  if (input_sample_rate != TARGET_SAMPLE_RATE) {
    printf("Input sample rate (%d Hz) != target rate (%d Hz), resampling input...\n", 
           input_sample_rate, TARGET_SAMPLE_RATE);
    input_resampler = init_resample_context(input_sample_rate, TARGET_SAMPLE_RATE, 1);
    if (!input_resampler) {
      fprintf(stderr, "Failed to initialize input resampler\n");
      fclose(f1);
      fclose(fout);
      rnnoise_destroy(st);
      return 1;
    }
  }
  
  if (output_sample_rate != TARGET_SAMPLE_RATE) {
    printf("Output sample rate (%d Hz) != target rate (%d Hz), resampling output...\n", 
           output_sample_rate, TARGET_SAMPLE_RATE);
    output_resampler = init_resample_context(TARGET_SAMPLE_RATE, output_sample_rate, 1);
    if (!output_resampler) {
      fprintf(stderr, "Failed to initialize output resampler\n");
      fclose(f1);
      fclose(fout);
      rnnoise_destroy(st);
      free_resample_context(input_resampler);
      return 1;
    }
  }
  
  // Write WAV output header (write placeholder, update later)
  if (!write_wav_header(fout, output_sample_rate, 1, 0)) { // 0 samples for now
    fprintf(stderr, "Error writing WAV header to output file\n");
    fclose(f1);
    fclose(fout);
    rnnoise_destroy(st);
    free_resample_context(input_resampler);
    free_resample_context(output_resampler);
    return 1;
  }
  
  printf("Processing audio...\n");
  
  int frame_idx = 0;
  while (1) {
    // Use dynamic frame size based on sample rate
    int current_frame_size = input_frame_size;
    short *tmp = malloc(input_frame_size * sizeof(short));
    int samples_read;
    
    // Read mono data
    samples_read = fread(tmp, sizeof(short), input_frame_size, f1);
    
    if (samples_read == 0) {
      free(tmp);
      break;
    }
    
    // If we read less than expected samples, pad with zeros
    for (i = samples_read; i < input_frame_size; i++) {
      tmp[i] = 0;
    }
    
    // Convert to float (pad or truncate to FRAME_SIZE for RNNoise)
    for (i = 0; i < input_frame_size; i++) {
      x[i] = tmp[i];
    }
    
    printf("frame_idx: %d, frame_size: %d\n", frame_idx, input_frame_size);
    
    // Resample input to 48kHz if needed
    if (input_resampler) {
      float temp_buffer[FRAME_SIZE * 4];
      memset(temp_buffer, 0, FRAME_SIZE * 4 * sizeof(float));
      int temp_frames = FRAME_SIZE * 4;
      if (!resample_audio(input_resampler, x, input_frame_size, temp_buffer, &temp_frames)) {
        fprintf(stderr, "Input resampling failed\n");
        free(tmp);
        break;
      }
      // Copy resampled data back to x
      memcpy(x, temp_buffer, FRAME_SIZE * sizeof(float));
    }
    
    // Process with RNNoise
    rnnoise_process_frame(st, x, x);
    frame_idx++;
    
    // Resample output back to original rate if needed
    memset(y, 0, FRAME_SIZE * sizeof(float));
    if (output_resampler) {
      float temp_buffer[FRAME_SIZE * 4];
      memset(temp_buffer, 0, FRAME_SIZE * 4 * sizeof(float));
      int temp_frames = FRAME_SIZE * 4;
      if (!resample_audio(output_resampler, x, FRAME_SIZE, temp_buffer, &temp_frames)) {
        fprintf(stderr, "Output resampling failed\n");
        free(tmp);
        break;
      }
      // Copy resampled data to y
      memcpy(y, temp_buffer, output_frame_size * sizeof(float));
    } else {
      // No resampling needed, copy x to y
      memcpy(y, x, output_frame_size * sizeof(float));
    }
    
    // Convert back to short (use current_frame_size for output)
    for (i = 0; i < output_frame_size; i++) {
      if (y[i] > 32767) y[i] = 32767;
      if (y[i] < -32768) y[i] = -32768;
      tmp[i] = (short)y[i];
    }
    
    // Write output
    fwrite(tmp, sizeof(short), samples_read, fout);
    output_samples_count += samples_read;
    
    free(tmp);
  }
  
  // Update WAV header with correct file size
  update_wav_header_size(fout, output_samples_count, 1);
  
  printf("Processing complete. Output samples: %d\n", output_samples_count);
  
  // Cleanup
  rnnoise_destroy(st);
  free_resample_context(input_resampler);
  free_resample_context(output_resampler);
  fclose(f1);
  fclose(fout);
  return 0;
}
