#include <stdlib.h>
#include <stdio.h>
#include "bitmap.h"

#define XSIZE 2560 // Size of before image
#define YSIZE 2048

#define NEW_XSIZE (XSIZE*2)
#define NEW_YSIZE (YSIZE*2)


void resizeImage(const uchar* beforeImage, int oldWidth, int oldHeight, int newWidth, int newHeight, uchar* newImage);


int main(void)
{
	uchar *image = calloc(XSIZE * YSIZE * 3, 1); // Three uchars per pixel (RGB)
	readbmp("before.bmp", image);

	// Alter the image here
	uchar *newImage = calloc(NEW_XSIZE * NEW_YSIZE * 3, 1); // Three uchars per pixel (RGB)
	resizeImage(image, XSIZE, YSIZE, NEW_XSIZE, NEW_YSIZE, newImage);

	savebmp("after.bmp", newImage, NEW_XSIZE, NEW_YSIZE);

	free(image);
	free(newImage);
	return 0;
}


void invertColors(const uchar* beforeImage, uchar* newImage) 
{
	return;
}


void resizeImage(const uchar* beforeImage, int oldWidth, int oldHeight, int newWidth, int newHeight, uchar* newImage)
{
	// Resize image using nearest neighbor interpolation
	for (int y = 0; y < newHeight; y++) {
		for (int x = 0; x < newWidth; x++) {
			int oldX = x * oldWidth / newWidth;
			int oldY = y * oldHeight / newHeight;

			for (int c = 0; c < 3; c++) {
				newImage[(y * newWidth + x) * 3 + c] = beforeImage[(oldY * oldWidth + oldX) * 3 + c];
			}
		}
	}
}
