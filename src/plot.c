//#include <plplot/plplot.h>
#include <math.h>
#include <unistd.h>
#include <stdio.h>


int main()
{
	FILE *pipe = popen("gnuplot -persist", "w");
	float freq = 1;
	int i;
	for (i = 0; i < 1000; i++) {
		fprintf(pipe, "plot sin(%f*x)\n", freq);
		freq += .05;
	}
	pclose(pipe);
	return 0;
}

