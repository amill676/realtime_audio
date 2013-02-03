/***************************************************************
 * @file    audio_tools.c
 * @group   src
 * @brief   Implements functions delcared in audio_tools.h
 * @author  Adam Miller
 *
 * 9/15/12
 **************************************************************/

#include "audio_tools.h"

void fatal(char s[]) {
    fprintf(stderr, "ERROR: %s\n", s);
    exit(1);
}


void fatal_terminate(char s[])
{
    Pa_Terminate();
    fatal(s);
}


void pa_fatal(char s[], PaError err)
{
    Pa_Terminate(); // MUST CALL TO AVOID AUDIO DEVICE PROBLEMS
    fprintf(stderr, "ERROR: %s\n", s);
    fprintf(stderr, "CODE: %d\t MESSAGE: %s\n", 
                    err, Pa_GetErrorText(err));
    exit(1);
}
