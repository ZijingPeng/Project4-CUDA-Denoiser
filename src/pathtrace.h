#pragma once

#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(int frame, int iteration);
void showGBuffer(uchar4* pbo);
void showImage(uchar4* pbo, int iter);
void denoise(uchar4* pbo, int iter, int filter_size, float cw, float nw, float pw);
void showDenoisedImage(uchar4* pbo);
