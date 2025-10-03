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
#include "rnnoise.h"

#ifdef _WIN32
#define strcasecmp _stricmp
#endif

#define FRAME_SIZE 480

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

int main(int argc, char **argv) {
  int i;
  int first = 1;
  float x[FRAME_SIZE];
  FILE *f1, *fout;
  DenoiseState *st;
  WavHeader input_header;
  int input_samples_count = 0;
  int output_samples_count = 0;
  int is_input_wav, is_output_wav;
  
  st = rnnoise_create(NULL);
  if (argc!=3) {
    fprintf(stderr, "usage: %s <noisy speech> <output denoised>\n", argv[0]);
    fprintf(stderr, "Supports both raw PCM (16-bit, mono) and WAV formats\n");
    return 1;
  }
  
  // Check file formats
  is_input_wav = is_wav_file(argv[1]);
  is_output_wav = is_wav_file(argv[2]);
  
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
  
  // Handle WAV input
  if (is_input_wav) {
    if (!read_wav_header(f1, &input_header, &input_samples_count)) {
      fprintf(stderr, "Error reading WAV header from input file\n");
      fclose(f1);
      fclose(fout);
      rnnoise_destroy(st);
      return 1;
    }
    
    // Check if input is mono (RNNoise works on mono audio)
    if (input_header.num_channels != 1) {
      fprintf(stderr, "Warning: Input has %d channels, RNNoise works on mono audio.\n", input_header.num_channels);
      fprintf(stderr, "Only the first channel will be processed.\n");
    }
  } else {
    printf("Processing raw PCM file (assuming 16-bit, mono, 48kHz)\n");
  }
  
  // Handle WAV output header (write placeholder, update later)
  if (is_output_wav) {
    int sample_rate = is_input_wav ? input_header.sample_rate : 48000;
    if (!write_wav_header(fout, sample_rate, 1, 0)) { // 0 samples for now
      fprintf(stderr, "Error writing WAV header to output file\n");
      fclose(f1);
      fclose(fout);
      rnnoise_destroy(st);
      return 1;
    }
  }
  
  printf("Processing audio...\n");
  
  while (1) {
    short tmp[FRAME_SIZE];
    int samples_read;
    
    if (is_input_wav && input_header.num_channels > 1) {
      // Read multi-channel data but only process first channel
      short multi_channel_tmp[FRAME_SIZE * input_header.num_channels];
      samples_read = fread(multi_channel_tmp, sizeof(short), FRAME_SIZE * input_header.num_channels, f1);
      samples_read /= input_header.num_channels;
      
      // Extract first channel
      for (i = 0; i < samples_read; i++) {
        tmp[i] = multi_channel_tmp[i * input_header.num_channels];
      }
    } else {
      // Read mono data
      samples_read = fread(tmp, sizeof(short), FRAME_SIZE, f1);
    }
    
    if (samples_read == 0) break;
    
    // If we read less than FRAME_SIZE samples, pad with zeros
    for (i = samples_read; i < FRAME_SIZE; i++) {
      tmp[i] = 0;
    }
    
    // Convert to float
    for (i = 0; i < FRAME_SIZE; i++) {
      x[i] = tmp[i];
    }
    
    // Process with RNNoise
    rnnoise_process_frame(st, x, x);
    
    // Convert back to short
    for (i = 0; i < FRAME_SIZE; i++) {
      tmp[i] = (short)x[i];
    }
    
    // Write output (skip first frame as in original)
    if (!first) {
      fwrite(tmp, sizeof(short), FRAME_SIZE, fout);
      output_samples_count += FRAME_SIZE;
    }
    first = 0;
  }
  
  // Update WAV header with correct file size
  if (is_output_wav) {
    update_wav_header_size(fout, output_samples_count, 1);
  }
  
  printf("Processing complete. Output samples: %d\n", output_samples_count);
  
  rnnoise_destroy(st);
  fclose(f1);
  fclose(fout);
  return 0;
}
