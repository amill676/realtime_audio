/**
 * @file realtimestft.h
 *
 * @author Adam Miller
 */

#ifndef REALTIMESTFT_H
#define REALTIMESTFT_H

// vDSP.h in /System/Library/Frameworks/vecLib.framework/Versions/A/Headers
#include <vDSP.h>
#include <Accelerate/Accelerate.h>

/** Enumeration of error codes */
typedef enum {
	STFT_OK,
	STFT_FAILED_MALLOC,
	STFT_INVALID_HOPSIZE,
	STFT_INVALID_WINDOWSIZE,
	STFT_INVALID_DFTLEN,
	STFT_INVALID_NUM_CHANNELS,
	STFT_NULL_PARAMETER,
	STFT_FFTSETUP_ERROR
} stft_error;


/**
 * Struct containing all necessary members for perfomring
 * the realtime STFT
 */
typedef struct realtimeSTFT {
	
	int dft_log2n;			///< log of DFT length
	int num_channels;		///< number of channels in audio
	
	int window_len;		///< log of window lenght
	int hop_size;			///< log of hope size
	float *window_buf;		///< used to hold window function

	float *in_buf;		///< buffer for input data 
	int curr_in_ind;			///< tracks current window frame

	float *out_buf;
	int curr_out_ind;

	int num_dfts;					///< Number of dfts per channel
	DSPSplitComplex *dfts;			///< array of dfts 

} realtimeSTFT;


int createRealtimeSTFT( realtimeSTFT *, int, int, int, int );
int destroyRealtimeSTFT( realtimeSTFT * );
int performSTFT( realtimeSTFT *, float *);
int performISTFT( realtimeSTFT *, float *);

#endif
