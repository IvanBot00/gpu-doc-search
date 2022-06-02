#ifndef __FILEH__
#define __FILEH__

#include <sys/time.h>

typedef struct {
    struct timeval startTime;
    struct timeval endTime;
} Timer;

#ifdef __cplusplus
extern "C" {
#endif
void verify_bm25(const float*, const unsigned*, const unsigned*, unsigned int, unsigned int);
void verify_df(const unsigned*, const unsigned*,  unsigned int, unsigned int);
void startTime(Timer* timer);
void stopTime(Timer* timer);
float elapsedTime(Timer timer);
#ifdef __cplusplus
}
#endif

#endif
