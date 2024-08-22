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
	uchar *invertedColorImage = calloc(XSIZE * YSIZE * 3, 1); // Three uchars per pixel (RGB)
	invertColors(image, XSIZE, YSIZE, invertedColorImage); // Invert colors

	uchar *newImage = calloc(NEW_XSIZE * NEW_YSIZE * 3, 1); // New bitmap of twice the original size
	resizeImage(invertedColorImage, XSIZE, YSIZE, NEW_XSIZE, NEW_YSIZE, newImage); // Resize image using nearest neighbor interpolation

	savebmp("after.bmp", newImage, NEW_XSIZE, NEW_YSIZE); // Save image

	free(image);
	free(newImage);
	return 0;
}


void invertColors(const uchar* beforeImage, int width, int height, uchar* newImage) 
{
	// Iterate through image
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			for (int c = 0; c < 3; c++) {
				int index = (y * width + x) * 3 + c;
				newImage[index] = 255 - beforeImage[index];
			}
		}
	}
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
