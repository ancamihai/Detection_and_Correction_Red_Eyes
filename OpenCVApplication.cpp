// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>

wchar_t* projectPath;

void testOpenImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image", src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName) == 0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName, "bmp");
	while (fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(), src);
		if (waitKey() == 27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	_wchdir(projectPath);

	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", IMREAD_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, COLOR_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Accessing individual pixels in an 8 bits/pixel image
		// Inefficient way -> slow
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				uchar neg = 255 - val;
				dst.at<uchar>(i, j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testNegativeImageFast()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// The fastest approach of accessing the pixels -> using pointers
		uchar* lpSrc = src.data;
		uchar* lpDst = dst.data;
		int w = (int)src.step; // no dword alignment is done !!!
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i * w + j];
				lpDst[i * w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);

		// Accessing individual pixels in a RGB 24 bits/pixel image
		// Inefficient way -> slow
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i, j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i, j) = (r + g + b) / 3;
			}
		}

		imshow("input image", src);
		imshow("gray image", dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// HSV components
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// Defining pointers to each matrix (8 bits/pixels) of the individual components H, S, V 
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, COLOR_BGR2HSV);

		// Defining the pointer to the HSV image matrix (24 bits/pixel)
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int hi = i * width * 3 + j * 3;
				int gi = i * width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;	// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1, dst2;
		//without interpolation
		resizeImg(src, dst1, 320, false);
		//with interpolation
		resizeImg(src, dst2, 320, true);
		imshow("input image", src);
		imshow("resized image (without interpolation)", dst1);
		imshow("resized image (with interpolation)", dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src, dst, gauss;
		src = imread(fname, IMREAD_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int)k * pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss, dst, pL, pH, 3);
		imshow("input image", src);
		imshow("canny", dst);
		waitKey();
	}
}

void testVideoSequence()
{
	_wchdir(projectPath);

	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}

	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
		Canny(grayFrame, edges, 40, 100, 3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = waitKey(100);  // waits 100ms and advances to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n");
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	_wchdir(projectPath);

	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];

	// video resolution
	Size capS = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;

		imshow(WIN_SRC, frame);

		c = waitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115) { //'s' pressed - snap the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess)
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == EVENT_LBUTTONDOWN)
	{
		printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
			x, y,
			(int)(*src).at<Vec3b>(y, x)[2],
			(int)(*src).at<Vec3b>(y, x)[1],
			(int)(*src).at<Vec3b>(y, x)[0]);
	}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / Image Processing)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

void rednessDetectionWithPredefinedFunctions(Mat srcImg, float percentage)
{
	int height = srcImg.rows;
	int width = srcImg.cols;
	Mat dst = Mat(height, width, CV_8UC1);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			Vec3b pixel = srcImg.at< Vec3b>(i, j);
			unsigned char B = pixel[0];
			unsigned char G = pixel[1];
			unsigned char R = pixel[2];

			double Redness;
			if (R > 0)
			{
				Redness = max(0.0, (2 * R - (G + B)) / R) * max(0.0, (2 * R - (G + B)) / R);
			}
			else
			{
				Redness = 0.0;
			}

			if (Redness > 0.999999995)
			{
				dst.at<uchar>(i, j) = 255;
			}
			else
			{
				dst.at<uchar>(i, j) = 0;
			}
		}
	}

	imshow("redness", dst);
	std::vector<Vec3f> circles;
	double dp = 1;
	double minDist = 20;   // Minimum distance between detected centers
	double param1 = 50;     // Upper threshold for the internal Canny edge detector
	double param2 = 10;     // Threshold for center detection
	float minRadius = dst.rows < 200 ? 2.75 : dst.rows / (200 * percentage);
	int maxRadius = dst.rows < 100 ? dst.rows : dst.rows / (27 * percentage);

	HoughCircles(dst, circles, HOUGH_GRADIENT, dp, minDist,
		param1, param2, minRadius, maxRadius);

	Point center(dst.cols / 2, dst.rows / 2);

	// Sort circles based on their distance to the center of the image
	sort(circles.begin(), circles.end(), [&](const Vec3f& a, const Vec3f& b) {
		float distA = norm(Point(a[0], a[1]) - center);
		float distB = norm(Point(b[0], b[1]) - center);
		return distA < distB;
		});

	std::vector<Vec3f> firstTwoCircles(circles.begin(), circles.begin() + min(2, int(circles.size())));

	Mat result = Mat::zeros(dst.size(), CV_8UC3);
	for (const auto& circle1 : firstTwoCircles) {
		Point center(cvRound(circle1[0]), cvRound(circle1[1]));
		int radius = cvRound(circle1[2]);
		circle(srcImg, center, radius, Scalar(0, 255, 0), -1);
	}

	imshow("Red eyes detection", srcImg);
	waitKey();
}

boolean isInside(Mat img, int i, int j)
{
	int height = img.rows;
	int width = img.cols;

	if (i < 0)
	{
		return false;
	}

	if (i >= height)
	{
		return false;
	}

	if (j < 0)
	{
		return false;
	}

	if (j >= width)
	{
		return false;
	}

	return true;
}

void rednessDetection(Mat srcImg, int imgSize)
{
	int height = srcImg.rows;
	int width = srcImg.cols;
	Mat dst = Mat(height, width, CV_8UC1);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			Vec3b pixel = srcImg.at< Vec3b>(i, j);
			unsigned char B = pixel[0];
			unsigned char G = pixel[1];
			unsigned char R = pixel[2];

			double Redness;
			if (R > 0)
			{
				Redness = max(0.0, (2 * R - (G + B)) / R) * max(0.0, (2 * R - (G + B)) / R);
			}
			else
			{
				Redness = 0.0;
			}

			if (Redness > 0.999999995)
			{
				dst.at<uchar>(i, j) = 255;
			}
			else
			{
				dst.at<uchar>(i, j) = 0;
			}
		}
	}

	Mat dst_dilation = dst.clone();

	int di[4] = { -1,0,1,0 };
	int dj[4] = { 0,-1,0,1 };


	for (int l = 0; l < 2; l++)
	{
		Mat dst1 = Mat(height, width, CV_8UC1, cv::Scalar(0));

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (dst_dilation.at<uchar>(i, j) > 200)
				{
					dst1.at<uchar>(i, j) = 255;

					for (int k = 0; k < 4; k++)
					{
						if (isInside(dst_dilation, i + di[k], j + dj[k]))
						{
							dst1.at<uchar>(i + di[k], j + dj[k]) = 255;
						}
					}

				}
			}
		}

		dst_dilation = dst1.clone();

	}

	Mat dst_borders = dst_dilation.clone();

	Mat dst1 = Mat(height, width, CV_8UC1, cv::Scalar(0));

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (dst_dilation.at<uchar>(i, j) > 200)
			{
				bool isNotOk = false;

				for (int k = 0; k < 4; k++)
				{
					if (isInside(dst_dilation, i + di[k], j + dj[k]))
					{
						if (dst_dilation.at<uchar>(i + di[k], j + dj[k]) < 50)
						{
							isNotOk = true;
						}
					}

				}
				if (isNotOk == false)
				{
					dst1.at<uchar>(i, j) = 255;
				}
			}
		}
	}

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (dst_dilation.at<uchar>(i, j) > 200 && dst1.at<uchar>(i, j) == 0)
			{
				dst_borders.at<uchar>(i, j) = 255;
			}
			else
			{
				dst_borders.at<uchar>(i, j) = 0;
			}
		}
	}


	Mat dst_HOUGH = Mat::zeros(height, width, CV_8UC1);


	int minRadius;
	int maxRadius;
	int threshold = 140;

	if (imgSize < 50000)
	{
		minRadius = 5;
		maxRadius = 10;
		threshold = 120;
	}
	else if (imgSize < 150000)
	{
		minRadius = 6;
		maxRadius = 10;
	}
	else if (imgSize < 250000)
	{
		minRadius = 8;
		maxRadius = 20;
	}
	else 
	{
		minRadius = 10;
		maxRadius = 50;
	}

	std::vector<std::vector<std::vector<std::pair<int, int>>>> accum(height, std::vector<std::vector<std::pair<int, int>>>(width));
	std::vector<std::vector<int>> consideredPoints;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j <width; j++)
		{
			for (int r = minRadius; r <= maxRadius; r++)
			{
				accum[i][j].push_back(std::make_pair(r, 0));
			}
		}
	}

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (dst_borders.at<uchar>(i, j) > 200)
			{
				for (int r = minRadius; r <= maxRadius; r++)
				{
					for (int n = 0; n <= 360; ++n) {

						int a = i - r * sin(n * CV_PI / 180);
						int b = j - r * cos(n * CV_PI / 180);

						if (isInside(dst_borders, a, b))
						{
							accum[a][b][r - minRadius].second += 1;
						}

					}
				}
			}
		}
	}

	int Kernel = 3;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			int pixel = 0, temp = 0, x0 = 0, y0 = 0, r0 = 0;

			for (int y = -Kernel; y <= Kernel; y++) {
				for (int x = -Kernel; x <= -Kernel; x++) {
					for (int r = minRadius; r <= maxRadius; r++) {
						int tempY = i + y;
						int tempX = x + j;
						if (isInside(dst_borders,tempY, tempX)) {
							temp = accum[tempY][tempX][r-minRadius].second;
							if (temp > pixel) {
								pixel = temp;
								x0 = tempX;
								y0 = tempY;
								r0 = r;
							}
						}
					}
				}
			}

			if (pixel >threshold) {

				bool isDistantEnough = true;
				for (int n = 0; n < consideredPoints.size(); n++)
				{
					if ( sqrt( (x0 - consideredPoints[n][0]) * (x0 - consideredPoints[n][0]) + (y0 - consideredPoints[n][1]) * (y0 - consideredPoints[n][1])) <20)
					{
						isDistantEnough = false;
					}
				}

				if (isDistantEnough == true)
				{
					std::vector<int> currentPoint;
					currentPoint.push_back(x0);
					currentPoint.push_back(y0);
					currentPoint.push_back(r0);
					currentPoint.push_back(pixel);
					consideredPoints.push_back(currentPoint);
				}
			}
		}
	}


	Point center(dst.cols / 2, dst.rows / 2);

	sort(consideredPoints.begin(), consideredPoints.end(), [&](const std::vector<int>& a, const std::vector<int>& b) {
		float distA = norm(Point(a[0], a[1]) - center);
		float distB = norm(Point(b[0], b[1]) - center);
		return distA < distB;
		});

	Mat dst_HOUGH_blurred = dst_HOUGH.clone();

	if (consideredPoints.size() >= 2)
	{
		for (int i = 0; i <= 1; i++)
		{   
			
			int x0 = consideredPoints[i][0];
			int y0 = consideredPoints[i][1];
			int r0 = consideredPoints[i][2];

			for (int y = y0 - r0 - 3; y <= y0 + r0 + 3; ++y) {
				for (int x = x0 - r0 - 2; x <= x0 + r0 + 2; ++x) {
					if ((x - x0) * (x - x0) + (y - y0) * (y - y0) <= r0 * r0) {
						if (isInside(dst_borders, y, x)) {
							dst_HOUGH.at<uchar>(y, x) = 255;
						}
					}
					else if ((x - x0) * (x - x0) + (y - y0) * (y - y0) <= (r0 + 1) * (r0 + 1) && r0 >= 8)
					{
						if (isInside(dst_borders, y, x)) {
							dst_HOUGH.at<uchar>(y, x) = 255;
						}
					}
					else if ((x - x0) * (x - x0) + (y - y0) * (y - y0) <= (r0 + 2) * (r0 + 2) && r0 >=10)
					{
						if (isInside(dst_borders, y, x)) {
							dst_HOUGH.at<uchar>(y, x) = 255;
						}
					}
				}
			}
		}

		int size = 5;

		float standardDeviation = (float)size / 6.0;

		float cst = 1 / (2.0 * standardDeviation * standardDeviation * CV_PI);

		std::vector<float> gaussianKernel;

		for (int i = -(size / 2); i <= size / 2; i++)
		{
			for (int j = -(size / 2); j <= size / 2; j++)
			{
				float exponent = exp((float)-(((float)i * i + j * j) / (2.0 * standardDeviation * standardDeviation)));
				float value = cst * exponent;
				gaussianKernel.push_back(value);
			}
		}

		float sum = 0.0f;

		for (int i = 0; i < gaussianKernel.size(); i++)
		{
			sum += gaussianKernel[i];
		}

		int w = sqrt(gaussianKernel.size());
		int k = (w - 1) / 2;


		for (int i = k; i < height - k; i++)
		{
			for (int j = k; j < width - k; j++)
			{   
				float id = 0;
				for (int u = 0; u < w; u++)
				{
					for (int v = 0; v < w; v++)
					{
						id += gaussianKernel[w * u + v] * dst_HOUGH.at<uchar>(i + u - k, j + v - k);
					}
				}

				dst_HOUGH_blurred.at<uchar>(i, j) = static_cast<uchar>(id);

			}
		}


	}

	Mat red_correction = srcImg.clone();

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (dst_HOUGH_blurred.at<uchar>(i, j) >0)
			{
				Vec3b pixel = srcImg.at< Vec3b>(i, j);
				unsigned char B = pixel[0];
				unsigned char G = pixel[1];
				unsigned char R = pixel[2];

				unsigned char R_new = R - (dst_HOUGH_blurred.at<uchar>(i, j) / 255.0) * (R - min(G, B));

				unsigned char G_new = G;

				if (G > R_new)
				{
					G_new = (R_new + B) / 2;
				}

				unsigned char B_new = B;

				if (B > R_new)
				{
					B_new = (R_new + G) / 2;
				}

				Vec3b newPixel = Vec3b(B_new, G_new, R_new);

				red_correction.at<Vec3b>(i, j) = newPixel;

			}
			
		}
	}

	imshow("redness", dst);

	imshow("dilation applied", dst_dilation);

	imshow("extracting borders", dst_borders);

	imshow("hough", dst_HOUGH);

	imshow("hough_blurred", dst_HOUGH_blurred);

	imshow("red_correction", red_correction);

	waitKey(0);

}

cv::Point startPoint, endPoint;
bool isDragging = false;

void projectCallBackFunc(int event, int x, int y, int flags, void* param)
{
	if (event == cv::EVENT_LBUTTONDOWN) {
		startPoint = cv::Point(x, y);
		isDragging = true;
	}
	else if (event == cv::EVENT_MOUSEMOVE && isDragging == true) {
		cv::Mat tempImg = (*(cv::Mat*)param).clone();
		cv::rectangle(tempImg, startPoint, cv::Point(x, y), cv::Scalar(0, 255, 0), 2);
		imshow("Original Image", tempImg);
	}
	else if (event == cv::EVENT_LBUTTONUP && isDragging == true) {
		endPoint = cv::Point(x, y);
		isDragging = false;
		cv::Mat selectedImg = (*(cv::Mat*)param).clone();
		cv::rectangle(selectedImg, startPoint, endPoint, cv::Scalar(0, 255, 0), 2);
		imshow("Original Image", selectedImg);

		int minX = min(startPoint.x, endPoint.x);
		int minY = min(startPoint.y, endPoint.y);
		int maxX = max(startPoint.x, endPoint.x);
		int maxY = max(startPoint.y, endPoint.y);

		if (maxX - minX >= 20 && maxY - minY >= 20)

		{
			cv::Mat selectedArea = (*(cv::Mat*)param)(cv::Rect(minX, minY, maxX - minX, maxY - minY)).clone();

			imshow("Selected Area", selectedArea);

			//float percentage = (maxX - minX) * (maxY - minY) / (float)((*(cv::Mat*)param).rows * (*(cv::Mat*)param).cols);

			rednessDetection(selectedArea, (*(cv::Mat*)param).rows * (*(cv::Mat*)param).cols);
		}

	}
}


void project()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);

		//Create a window
		namedWindow("Original Image", 1);

		//set the callback function for any mouse event
		setMouseCallback("Original Image", projectCallBackFunc, &src);

		//show the image
		imshow("Original Image", src);

		// Wait until user press some key
		waitKey(0);
	}
}



int main()
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
	projectPath = _wgetcwd(0, 0);

	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative\n");
		printf(" 4 - Image negative (fast)\n");
		printf(" 5 - BGR->Gray\n");
		printf(" 6 - BGR->Gray (fast, save result to disk) \n");
		printf(" 7 - BGR->HSV\n");
		printf(" 8 - Resize image\n");
		printf(" 9 - Canny edge detection\n");
		printf(" 10 - Edges in a video sequence\n");
		printf(" 11 - Snap frame from live video\n");
		printf(" 12 - Mouse callback demo\n");
		printf(" 13 - Red eye detection and correction\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);
		switch (op)
		{
		case 1:
			testOpenImage();
			break;
		case 2:
			testOpenImagesFld();
			break;
		case 3:
			testNegativeImage();
			break;
		case 4:
			testNegativeImageFast();
			break;
		case 5:
			testColor2Gray();
			break;
		case 6:
			testImageOpenAndSave();
			break;
		case 7:
			testBGR2HSV();
			break;
		case 8:
			testResize();
			break;
		case 9:
			testCanny();
			break;
		case 10:
			testVideoSequence();
			break;
		case 11:
			testSnap();
			break;
		case 12:
			testMouseClick();
			break;
		case 13:
			project();
			break;
		}
	} while (op != 0);
	return 0;
}