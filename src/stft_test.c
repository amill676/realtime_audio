#include "realtimestft.h"
#include "stdio.h"
#include "vDSP.h"
#include <Accelerate/Accelerate.h>
#include "string.h"

#define DFT_LOG_LEN 3
#define DFT_LEN (1 << DFT_LOG_LEN)
#define WINDOW_LOG_LEN 3
#define WINDOW_LEN (1 << WINDOW_LOG_LEN)
#define HOP_LOG_N 3
#define N_CHANNELS 1
#define DFT_RADIX 2
#define USE_WINDOW 1
#define SAMPLE float
#define SEPARATOR "=========================================================\n"


void print_dft( float * reals, float * imags, int dft_len) {
	printf("reals:\n");
	for (int j = 0; j < dft_len/2; j++) {
		printf("%f\t", reals[j]);
	}
	printf("\nimags:\n");
	for (int j = 0; j < dft_len/2; j++) {
		printf("%f\t", imags[j]);
	}
	printf("\n\n");
}

void setup_dsp_split(DSPSplitComplex *split, int dft_len) {
	split->realp = (float *) malloc( sizeof(float) * dft_len/2);
	split->imagp = (float *) malloc( sizeof(float) * dft_len/2);
	if (split->realp == NULL || split->imagp == NULL) {
		fprintf(stderr, "Malloc error in setting up split\n");
		exit(1);
	}
}

void free_dsp_split(DSPSplitComplex *split) {
	free(split->realp);
	free(split->imagp);
}

void print_buffer( float * buf, int len) {
	for (int i = 0; i < len; i++) {
		printf("%f\t", buf[i]);
	}
	printf("\n");
}

int main() {

	realtimeSTFT stft;
	int err = createRealtimeSTFT(&stft, DFT_LOG_LEN, WINDOW_LOG_LEN, 
							 HOP_LOG_N, N_CHANNELS, USE_WINDOW, sizeof(SAMPLE));
	if (err != STFT_OK) {
		fprintf(stderr, "Error in setting up stft object\n");
		exit(1);
	}

	float data[WINDOW_LEN] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
	float *buf = (float *)malloc( WINDOW_LEN * sizeof(float));

	printf("%s", SEPARATOR);
	printf("Testing DFT of data\n");
	DSPSplitComplex split;
	setup_dsp_split(&split, DFT_LEN);
	FFTSetup fft_setup = vDSP_create_fftsetup(DFT_LOG_LEN, DFT_RADIX);
	if (fft_setup == NULL) {
		fprintf(stderr, "Problem creating fftsetup obj\n");
		exit(1);
	}
	// Copy into buffer for dft
	memcpy(buf, data, WINDOW_LEN * sizeof(float));
	/* Put into even-odd format */
	vDSP_ctoz( (DSPComplex *)buf, 2, &split, 1, DFT_LEN/2);
	/* Peform fft */
	vDSP_fft_zrip(fft_setup, &split, 1, DFT_LOG_LEN, kFFTDirection_Forward);
	print_dft(split.realp, split.imagp, DFT_LEN);
	/* Peform ifft */
	vDSP_fft_zrip(fft_setup, &split, 1, DFT_LOG_LEN, kFFTDirection_Inverse);
	/* Scale to compensate for inverse  */
	float scale = (float)(1.0/(2*DFT_LEN));
	vDSP_vsmul(split.realp, 1, &scale, split.realp, 1, DFT_LEN/2);
	vDSP_vsmul(split.imagp, 1, &scale, split.imagp, 1, DFT_LEN/2);
	/* Put back into regular form */
	vDSP_ztoc(&split, 1, (DSPComplex *)buf, 2, DFT_LEN/2);
	printf("IFFT result:\n");
	print_buffer(buf, WINDOW_LEN);


	printf("%s", SEPARATOR);
	printf("Testing STFT of {1,2,3,4,5,6,7,8}\n");
	performSTFT(&stft, data);
	for (int i = 0; i < stft.num_dfts; i++) {
		printf("DFT %d:\n", i);
		print_dft( stft.dfts[i].realp, stft.dfts[i].imagp, DFT_LEN);
	}

	float *out = (float *)malloc( sizeof(float) * WINDOW_LEN);
	performISTFT(&stft, out);
	printf("Result of istft:\n");
	for (int i = 0; i < WINDOW_LEN; i++) 
		printf("%f\t", out[i]);

	performSTFT(&stft, data);

	free_dsp_split(&split);
	free(buf);
	free(out);
	

	return 0;
}

	

