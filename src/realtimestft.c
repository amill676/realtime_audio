/**
 * @file realtimestftnew.c
 *
 * @author Adam Miller
 */

#include "realtimestft.h"
#include <math.h>
#include <string.h>

#define pi (3.14159265)

#define DFT_RADIX kFFTRadix2


/**
 * Sets up realtimeSTFT struct as required for given parameters
 * @param obj 			realtimeSTFT to setup
 * @param dft_logn		log2 of the dft size
 * @param window_logn	log2 of the window length
 * @param hop_logn		log2 of hop size
 * @param num_channels	number of channels in data
 * @return				0 for no error
 */
int	createRealtimeSTFT(	realtimeSTFT *obj, 
						int dft_logn, 
						int window_logn,
						int hop_logn,
						int num_channels,
						int use_window_fcn, 
						int data_size)
{
	int buffer_bytes;
	/* Determine data type -- 
	 * NOT ACTUALLY USED. If double format becomes supported for portaudio data,
	 * use this field to determine was size buffers should be and such. THen
	 * uncomment doubleRealtimeSTFT struct */
	if (data_size != 4 && data_size != 8) {
		return STFT_INVALID_DATA_SIZE;  /* Must be float or double */
	}

	/* Setup members of struct */
	if (dft_logn < 0) return STFT_INVALID_DFTLEN;
	if (dft_logn != window_logn) return STFT_INVALID_DFTLEN;
	if (window_logn < 0) return STFT_INVALID_WINDOWSIZE;
	if (num_channels <= 0) return STFT_INVALID_NUM_CHANNELS;
	if (obj == NULL) return STFT_NULL_PARAMETER;
	obj->dft_log2n = dft_logn;
	obj->window_len = (1 << window_logn);
	obj->num_channels = num_channels;
	obj->use_window_fcn = use_window_fcn;

	/* Check for valid hopsize. Assign if valid */
	if (hop_logn > window_logn) return STFT_INVALID_HOPSIZE;
	obj->hop_size = (1 << hop_logn);

	/* Setup window buffer now that parameters are known */
	obj->window_buf = (float *) malloc((obj->window_len)* sizeof(float));
	if (obj->window_buf == NULL) return STFT_FAILED_MALLOC;
	int n, N = obj->window_len;
	/* Put in hann window */
	for (n = 0; n < N; n++) {
		obj->window_buf[n] = .5 * (1 - cos(2*pi*n/(N-1)));
	}

	/* Setup in buffer 
	 * in buffer will be 2 window_lengths worth of data for each channel, where
	 * all of a channel's data is placed before the next channel's data */
	buffer_bytes = 2 * num_channels * (obj->window_len) * sizeof(float);
	obj->in_buf = (float *) malloc(buffer_bytes);
	if (obj->in_buf == NULL) return STFT_FAILED_MALLOC;
	memset(obj->in_buf, 0, buffer_bytes);
	obj->curr_in_ind = 0;

	/* Setup out buffer */
	buffer_bytes = 2 * num_channels * (obj->window_len) * sizeof(float);
	obj->out_buf = (float *) malloc(buffer_bytes);
	if (obj->out_buf == NULL) return STFT_FAILED_MALLOC;
	memset(obj->out_buf, 0, buffer_bytes);
	obj->curr_out_ind = 0;

	/* Setup dft array */
	obj->num_dfts = (1 << (window_logn - hop_logn)); // # of dfts
	obj->dfts = (DSPSplitComplex *) malloc(obj->num_dfts * num_channels *
									sizeof(DSPSplitComplex));
	if (obj->dfts == NULL) return STFT_FAILED_MALLOC;

	/* Setup each DSPSplitComplex struct in dfts */
	for (n = 0; n < obj->num_dfts*num_channels; n++) {
		obj->dfts[n].realp = malloc(sizeof(float) *(1 << (obj->dft_log2n-1)));
		if (obj->dfts[n].realp == NULL) return STFT_FAILED_MALLOC;
		obj->dfts[n].imagp = malloc(sizeof(float) *(1 << (obj->dft_log2n-1)));
		if (obj->dfts[n].imagp == NULL) return STFT_FAILED_MALLOC;
	}

	return STFT_OK;
}

/**
 * Destroys realtimeSTFT struct by freeing associated memory
 * @param obj		struct to free up
 */
int destroyRealtimeSTFT( realtimeSTFT *obj )
{
	/* Free window buffer */
	if (obj->window_buf) free(obj->window_buf);
	/* Free data buffers */
	if (obj->in_buf) free(obj->in_buf);
	if (obj->out_buf) free(obj->out_buf);
	/* Free dft array */
	if (obj->dfts) {
		int i;
		for (i=0; i < obj->num_dfts * obj->num_channels; i++) {
			if (obj->dfts[i].realp) free(obj->dfts[i].realp);
			if (obj->dfts[i].imagp) free(obj->dfts[i].imagp);
		}
		free(obj->dfts);
	}
	return STFT_OK;
}


/**
 * Fills up dft array of realtimeSTFT struct with DFTs of windowed
 * segments of given input data. 
 * @param obj		realtimeSTFT object containing necessary parameters
 * @param data_in	buffer containing input signal segment to be windowedd
 * 					and transformed
 */
int performSTFT( realtimeSTFT *obj, float *data_in )
{
	/* Check for null input object */
	if (obj == NULL || data_in == NULL) 
		return STFT_NULL_PARAMETER;

	int window_len = obj->window_len;

	/* Copy data into buffer */
	int i,j;
	for (j=0; j<obj->num_channels; j++) {
		for (i=0; i< window_len; i++) {
			obj->in_buf[j*2*obj->window_len + 
				((obj->curr_in_ind + i) % (2*window_len))] 
			= data_in[i*obj->num_channels + j];
		}
	}

	/* buffer to hold data for performing dft */
	float *dft_buf = (float *)malloc(window_len *sizeof(float)); 
	DSPSplitComplex *split;
	int ind; 
	int dft_len = (1 << obj->dft_log2n);
	/* FIXME consider storing this in the struct to avoid allocating every time */
	FFTSetup fft_setup = vDSP_create_fftsetup(obj->dft_log2n, DFT_RADIX);
	if (fft_setup == NULL) return STFT_FFTSETUP_ERROR;


	for (j=0; j < obj->num_channels; j++) {
		for (i=0; i < obj->num_dfts ; i++) {
			/* Start index in buffer for the current channel */
			int chan_base = window_len*2*j;

			/* Find index for beginning of current data window */
			ind = obj->curr_in_ind - (obj->num_dfts -1 - i)*obj->hop_size
											+ chan_base;
			/* wraparound in buffer	*/
			if (ind < chan_base) ind += 2*window_len; 

			/* Copy from data buffer to temp buffer for dft */
			if ( ind <= window_len + chan_base)
				memcpy(dft_buf, &obj->in_buf[ind], sizeof(float)*window_len);
			else {
				int cpy_num = (2*window_len + chan_base - ind);
				memcpy(dft_buf, &obj->in_buf[ind], sizeof(float) * cpy_num);
				memcpy(dft_buf + cpy_num, &obj->in_buf[chan_base], 
									sizeof(float) * (window_len - cpy_num));
			}

			/* Window the temp buffer 
			 * Use sqrt for double windowing */
			if (obj->use_window_fcn) {
				int n;
				for (n=0; n<window_len; n++)
					dft_buf[n] *= sqrt(obj->window_buf[n]);
			}

			split = &obj->dfts[j*obj->num_channels + i];
			/* Put into even-odd format */
			vDSP_ctoz( (DSPComplex *)dft_buf, 2, split, 1, dft_len/2); 
			/* Perform dft */
			vDSP_fft_zrip(fft_setup, split, 1, 
							obj->dft_log2n, kFFTDirection_Forward);

		} // end for
	}
	/* Update current index */
	obj->curr_in_ind = (obj->curr_in_ind + window_len) % (2*window_len);

	if (dft_buf) free(dft_buf);
	/* Deallocate fft_setup object */
	vDSP_destroy_fftsetup(fft_setup);

	return STFT_OK;
}


/**
 * Will perform the inverse DFT's on the DFT's present in the
 * realtimeSTFT obj and then will assemble the resulting signals
 * with the correct overlap and window to construct an output
 * signal in the time domain
 * @param obj		realtimeSTFT object with DFT's and parameters
 * @param data_out	output buffer to hold signal 
 */
int performISTFT( realtimeSTFT *obj, float *data_out)
{
	/* Check for NULL inputs */
	if (obj == NULL || data_out == NULL)
		return STFT_NULL_PARAMETER;

	int i, j, n;
	int dft_len = (1 << obj->dft_log2n);
	float *idft_buf = (float *)malloc(obj->window_len * sizeof(float));
	DSPSplitComplex *curr_dft;

	/* Setup for fft */
	FFTSetup fft_setup = vDSP_create_fftsetup(obj->dft_log2n, DFT_RADIX);
	if (fft_setup == NULL) return STFT_FFTSETUP_ERROR;

	for (j = 0; j < obj->num_channels; j++) {
		for (i = 0; i < obj->num_dfts; i++) {

			curr_dft = &obj->dfts[j*obj->num_channels + i];
			/* Base index in out buffer for current channel */
			int chan_base = 2*j*obj->window_len;
			
			/* Perform idft */
			vDSP_fft_zrip(fft_setup, curr_dft, 1, obj->dft_log2n, 
														kFFTDirection_Inverse);
			/* Scale to compensate for inverse DFT */
			float scale = (float)1.0/(2*dft_len);
			vDSP_vsmul(curr_dft->realp, 1, &scale, 
						curr_dft->realp, 1, dft_len/2);
			vDSP_vsmul(curr_dft->imagp, 1, &scale, 
						curr_dft->imagp, 1, dft_len/2);

			/* Put back into regular form */
			vDSP_ztoc(curr_dft, 1, (DSPComplex *)idft_buf, 2, dft_len/2);


			/* Clear out next frame so we have clean slate to add to */
			int ind;
			ind = (obj->curr_out_ind + obj->window_len) % (2*obj->window_len)
																+ chan_base;
			memset(&obj->out_buf[ind], 0, obj->window_len * sizeof(float));
			
			/* Add result into out_buf. Apply windowing if necessary. Two cases are
			 * given so the check for windowing doesn't occur on each iteration */
			if (obj->use_window_fcn) {
				for (n = 0; n < obj->window_len; n++) {
					ind = (obj->curr_out_ind + i*obj->hop_size + n) % 
										( 2 * obj->window_len ) + chan_base;
					obj->out_buf[ind] += sqrt(obj->window_buf[n])*idft_buf[n];
				}
			} else {
				for (n = 0; n < obj->window_len; n++) {
					ind = (obj->curr_out_ind + i*obj->hop_size + n) % 
										( 2 * obj->window_len ) + chan_base;
					obj->out_buf[ind] += idft_buf[n];
				}
			}

		} // end for i
	} // end for j


	/* Copy into output data buffer */
	for (j = 0; j < obj->num_channels; j++) {
		for (i = 0; i < obj->window_len; i++) {
			data_out[i*obj->num_channels + j] = 
				obj->out_buf[obj->curr_out_ind + 2*j*obj->window_len + i];
		}
	}

	/* Update current index */
	obj->curr_out_ind = (obj->curr_out_ind + obj->window_len) %
						(2 * obj->window_len);

	vDSP_destroy_fftsetup(fft_setup);
	if (idft_buf) free(idft_buf);
	
	return STFT_OK;
}
