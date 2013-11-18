#ifndef Stopwatch_h
#define Stopwatch_h

#include <time.h>

class Stopwatch {
private:
	clock_t start;
	clock_t _stopwatch() const
	{
		return clock();
		
	}
public:
	Stopwatch() { reset(); }
	void reset() { start = _stopwatch(); }
	double read() const {	return (_stopwatch() - start)/CLOCKS_PER_SEC; }
};

#endif
