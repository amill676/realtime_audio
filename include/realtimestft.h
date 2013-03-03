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
	STFT_OK = 0,
	STFT_FAILED_MALLOC = 1,
	STFT_INVALID_HOPSIZE = 2,
	STFT_INVALID_WINDOWSIZE = 3,
	STFT_INVALID_DFTLEN = 4,
	STFT_INVALID_NUM_CHANNELS = 5,
	STFT_NULL_PARAMETER = 6,
	STFT_FFTSETUP_ERROR = 7,
	STFT_INVALID_DATA_SIZE = 8
} stft_error;


/**
 * Struct containing all necessary members for perfomring
 * the realtime STFT
 */
typedef struct realtimeSTFT {
	
	int dft_log2n;			///< log of DFT length
	int num_channels;		///< number of channels in audio
	
	int use_window_fcn;	///< whether or not should use window fcn (that is not rect)
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

///**
// * Struct containing all necessary members for perfomring
// * the realtime STFT using double precision
// */
//typedef struct doubleRealtimeSTFT {
//	
//	int dft_log2n;			///< log of DFT length
//	int num_channels;		///< number of channels in audio
//	
//	int window_len;		///< log of window lenght
//	int hop_size;			///< log of hope size
//	double *window_buf;		///< used to hold window function
//
//	double *in_buf;		///< buffer for input data 
//	int curr_in_ind;			///< tracks current window frame
//
//	double *out_buf;
//	int curr_out_ind;
//
//	int num_dfts;					///< Number of dfts per channel
//	DSPDoubleSplitComplex *dfts;			///< array of dfts 
//
//} doubleRealtimeSTFT;


int createRealtimeSTFT( realtimeSTFT *, 
						int dft_logn, 
						int window_logn, 
						int hop_logn, 
						int num_channels,
						int use_window_fcn,
						int data_size );
int destroyRealtimeSTFT( realtimeSTFT * );
int performSTFT( realtimeSTFT *, float *);
int performISTFT( realtimeSTFT *, float *);

#endif
