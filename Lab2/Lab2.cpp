#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cmath>
#include "omp.h"
using namespace std; 
using namespace cv;

const int ThreadsNumber = 4;
const int repeats = 10;

typedef Vec3b Pixel;

void bgr2cmykThreads(Mat& img, vector<Mat>& cmyk)
{
	//Allocate cmyk to store 4 componets
	for (int i = 0; i < 4; i++)
	{
		cmyk.push_back(Mat(img.size(), CV_8UC1));
	}

	// Get rgb
	vector<Mat> rgb;
	split(img, rgb);

	//#pragma omp parallel num_threads(ThreadsNumber)
	//#pragma omp critical
	//	cout << "CMYK Thread: " << omp_get_thread_num() << " | " << this_thread::get_id() << endl;

		//rgb to cmyk
#pragma omp parallel for num_threads(ThreadsNumber)
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			float r = (int)rgb[2].at<uchar>(i, j) / 255.;
			float g = (int)rgb[1].at<uchar>(i, j) / 255.;
			float b = (int)rgb[0].at<uchar>(i, j) / 255.;
			float k = min(min(1 - r, 1 - g), 1 - b);

			//cmyk[0].at<uchar>(i, j) = (1 - r - k) / (1 - k) * 255.;
			//cmyk[1].at<uchar>(i, j) = (1 - g - k) / (1 - k) * 255.;
			cmyk[2].at<uchar>(i, j) = (1 - b - k) / (1 - k) * 255.;
			//cmyk[3].at<uchar>(i, j) = k * 255.;
		}
	}
}
void modifyColorsBlueThreads(const Mat& imageBGR, Mat& imageResult)
{
	//cout << "\nFilter Blue is start";

	imageBGR.copyTo(imageResult);

	uchar B, G, R = 0;

//#pragma omp parallel num_threads(ThreadsNumber)
//	#pragma omp critical
//		cout << "BLUE Thread: " << omp_get_thread_num() << " | " << this_thread::get_id() << endl;	

#pragma omp parallel for num_threads(ThreadsNumber) private (B,G,R)
	for (int i = 0; i < imageBGR.rows; i++)
	{
		for (int j = 0; j < imageBGR.cols; j++)
		{
			B = imageBGR.at<Vec3b>(i, j)[0];
			G = imageBGR.at<Vec3b>(i, j)[1];
			R = imageBGR.at<Vec3b>(i, j)[2];
			imageResult.at<Vec3b>(i, j)[0] = B - (G + B) / 2;
		}
	}
	//imshow("Modified Blue", imageResult);  waitKey(0);

	//cout << "\nFilter Blue is end";
}
void modifyColorsYellowThreads(const Mat& imageBGR, Mat& imageResult)
{
	//cout << "\nFilter Yellow is start";

	imageBGR.copyTo(imageResult);

	vector<Mat> imageCMYK;
	bgr2cmykThreads(imageResult, imageCMYK);

	uchar B, G, R = 0;

//#pragma omp parallel num_threads(ThreadsNumber)
//	#pragma omp critical
//		cout << "YELLOW Thread: " << omp_get_thread_num() << " | " << this_thread::get_id() << endl;

#pragma omp parallel for num_threads(ThreadsNumber) private (B,G,R)
	for (int i = 0; i < imageBGR.rows; i++)
	{
		for (int j = 0; j < imageBGR.cols; j++)
		{
			if (imageCMYK[2].at<uchar>(i, j))
			{
				B = imageBGR.at<Vec3b>(i, j)[0];
				G = imageBGR.at<Vec3b>(i, j)[1];
				R = imageBGR.at<Vec3b>(i, j)[2];
				imageResult.at<Vec3b>(i, j) = R + G - 2 * (abs(R - G) + B);
			}
		}
	}
	//imshow("Modified Yellow", imageResult);  waitKey(0);

	//cout << "\nFilter Yellow is end";
}
void modifyColorsThreads(const Mat& imageBGR, Mat& imageResult)
{
	//cout << "\nFilter is start";

	modifyColorsBlueThreads(imageBGR, imageResult);
	modifyColorsYellowThreads(imageResult, imageResult);

	//cout << "\nFilter is end";
}

void modifyColorsBlueThreadsForEach(const Mat& imageBGR, Mat& imageResult)
{
	//cout << "\nFilter Blue is start";

	imageBGR.copyTo(imageResult);


	imageResult.forEach<Pixel>
		(
			[](Pixel& pixel, const int* position) -> void
			{
				pixel[0] = pixel[0] - (pixel[1] + pixel[0]) / 2;
			}
	);

	/*uchar B, G, R = 0;
	parallel_for_(Range(0, imageBGR.rows), [&](const Range& range)
		{
			for (int i = range.start; i < range.end; i++)
			{
				for (int j = 0; j < imageBGR.cols; j++)
				{
					B = imageBGR.at<Vec3b>(i, j)[0];
					G = imageBGR.at<Vec3b>(i, j)[1];
					R = imageBGR.at<Vec3b>(i, j)[2];
					imageResult.at<Vec3b>(i, j)[0] = B - (G + B) / 2;
				}
			}
			cout << "BLUE I am Thread: " << this_thread::get_id() << endl;
		}, 2);*/
		//imshow("Modified Blue", imageResult);  waitKey(0);

		//cout << "\nFilter Blue is end";
}
void modifyColorsYellowThreadsForEach(const Mat& imageBGR, Mat& imageResult)
{
	//cout << "\nFilter Yellow is start";

	imageBGR.copyTo(imageResult);

	vector<Mat> imageCMYK;
	bgr2cmykThreads(imageResult, imageCMYK);

	imageResult.forEach<Pixel>
		(
			[&imageCMYK, &imageBGR](Pixel& pixel, const int* position) -> void
			{
				if (imageCMYK[2].at<uchar>(position))
					pixel = imageBGR.at<Vec3b>(position)[2] + imageBGR.at<Vec3b>(position)[1] - 2 * (abs(imageBGR.at<Vec3b>(position)[2] - imageBGR.at<Vec3b>(position)[1]) + imageBGR.at<Vec3b>(position)[0]);
					//pixel = pixel[2] + pixel[1] - 2 * (abs(pixel[2] - pixel[1]) + pixel[0]);
			}
	);

	//uchar B, G, R = 0;
	//parallel_for_(Range(0, imageBGR.rows), [&](const Range& range)
	//	{
	//		for (int i = range.start; i < range.end; i++)
	//		{
	//			for (int j = 0; j < imageBGR.cols; j++)
	//			{
	//				if (imageCMYK[2].at<uchar>(i, j))
	//				{
	//					B = imageBGR.at<Vec3b>(i, j)[0];
	//					G = imageBGR.at<Vec3b>(i, j)[1];
	//					R = imageBGR.at<Vec3b>(i, j)[2];
	//					imageResult.at<Vec3b>(i, j) = R + G - 2 * (abs(R - G) + B);
	//				}
	//			}
	//		}
	//		//cout << "YELLOW I am Thread: " << this_thread::get_id() << endl;
	//	}, 6);
	//imshow("Modified Yellow", imageResult);  waitKey(0);

	//cout << "\nFilter Yellow is end";
}
void modifyColorsThreadsForEach(const Mat& imageBGR, Mat& imageResult)
{
	//cout << "\nFilter is start";

	modifyColorsBlueThreadsForEach(imageBGR, imageResult);
	modifyColorsYellowThreadsForEach(imageResult, imageResult);

	//cout << "\nFilter is end";
}

void bgr2cmyk(Mat& img, vector<Mat>& cmyk)
{
	//Allocate cmyk to store 4 componets
	for (int i = 0; i < 4; i++)
	{
		cmyk.push_back(Mat(img.size(), CV_8UC1));
	}

	// Get rgb
	vector<Mat> rgb;
	split(img, rgb);

	//rgb to cmyk
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			float r = (int)rgb[2].at<uchar>(i, j) / 255.;
			float g = (int)rgb[1].at<uchar>(i, j) / 255.;
			float b = (int)rgb[0].at<uchar>(i, j) / 255.;
			float k = min(min(1 - r, 1 - g), 1 - b);

			//cmyk[0].at<uchar>(i, j) = (1 - r - k) / (1 - k) * 255.;
			//cmyk[1].at<uchar>(i, j) = (1 - g - k) / (1 - k) * 255.;
			cmyk[2].at<uchar>(i, j) = (1 - b - k) / (1 - k) * 255.;
			//cmyk[3].at<uchar>(i, j) = k * 255.;
		}
	}
}
void modifyColorsBlue(const Mat& imageBGR, Mat& imageResult)
{
	//cout << "\nFilter Blue is start";

	imageBGR.copyTo(imageResult);

	uchar B, G, R= 0;
	for (int i = 0; i < imageBGR.rows; i++)
	{
		for (int j = 0; j < imageBGR.cols; j++)
		{
			B = imageBGR.at<Vec3b>(i, j)[0];
			G = imageBGR.at<Vec3b>(i, j)[1];
			R = imageBGR.at<Vec3b>(i, j)[2];
			imageResult.at<Vec3b>(i, j)[0] = B - (G + B)/2;
		}
	}
	//imshow("Modified Blue", imageResult);  waitKey(0);

	//cout << "\nFilter Blue is end";
}
void modifyColorsYellow(const Mat& imageBGR, Mat& imageResult)
{
	//cout << "\nFilter Yellow is start";

	imageBGR.copyTo(imageResult);

	vector<Mat> imageCMYK;
	bgr2cmyk(imageResult, imageCMYK);
	
	uchar B, G, R = 0;
	for (int i = 0; i < imageBGR.rows; i++)
	{
		for (int j = 0; j < imageBGR.cols; j++)
		{
			if (imageCMYK[2].at<uchar>(i, j))
			{
				B = imageBGR.at<Vec3b>(i, j)[0];
				G = imageBGR.at<Vec3b>(i, j)[1];
				R = imageBGR.at<Vec3b>(i, j)[2];
				imageResult.at<Vec3b>(i, j) = R + G - 2 * (abs(R - G) + B);
			}
		}
	}
	//imshow("Modified Yellow", imageResult);  waitKey(0);

	//cout << "\nFilter Yellow is end";
}
void modifyColors(const Mat& imageBGR, Mat& imageResult)
{
	//cout << "\nFilter is start";

	modifyColorsBlue(imageBGR, imageResult);
	modifyColorsYellow(imageResult, imageResult);

	//cout << "\nFilter is end";
}

int main()
{
	//Загрузка изображений
	//Mat image1024x768  = imread("E:/Study/ТПУ МАГИСТРАТУРА/1 Семестр/Системы параллельной обработки данных/Лаб 2/Lab2/Data/1024x768.jpg");
	//Mat image1280x960  = imread("E:/Study/ТПУ МАГИСТРАТУРА/1 Семестр/Системы параллельной обработки данных/Лаб 2/Lab2/Data/1280x960.jpg");
	Mat image2048x1536 = imread("F:\\test2048.jpeg");
	//Mat image7680x4320 = imread("E:/Study/ТПУ МАГИСТРАТУРА/1 Семестр/Системы параллельной обработки данных/Лаб 2/Lab2/Data/7680x4320.jpg");
	//Проверка на ошибки загрузки изображений
	if (image2048x1536.empty())
	{
		cout << "\nImages Not Found!!!" << endl;
		waitKey(0);
		return -1;
	}
	//Показать загруженные изображения
	//imshow("Image 1024x768", image1024x768);  waitKey(0);
	//imshow("Image 1280x960", image1280x960);  waitKey(0);
	//imshow("Image 2048x1536", image2048x1536); waitKey(0);
	//imshow("Image 7680x4320", image7680x4320); waitKey(0);

	Mat image2048x1536Modifed;

	vector<chrono::duration<float>> severalThreadDuration;
	vector<chrono::duration<float>> oneThreadDuration;

	chrono::steady_clock::time_point start;
	chrono::steady_clock::time_point end;

	cout << "||";
	for (int i = 0; i < repeats; i++)
		cout << "=";
	cout << "||" << endl << "||";
	for (int n = 0; n < repeats; n++)
	{
		cout << "=";

		// =============== Многопоточная модификация каналов ===============
		start = chrono::high_resolution_clock::now(); //Точка для начала счета времени

		//modifyColorsThreads(image1024x768, image1024x768Modifed);
		//modifyColorsThreads(image1280x960, image1280x960Modifed);
		modifyColorsThreads(image2048x1536, image2048x1536Modifed);
		//modifyColorsThreads(image7680x4320, image7680x4320Modifed);

		//modifyColorsBlueThreads(image7680x4320, image7680x4320Modifed);
		//modifyColorsYellowThreads(image7680x4320, image7680x4320Modifed);

		//modifyColorsThreadsForEach(image1024x768, image1024x768Modifed);
		//modifyColorsThreadsForEach(image1280x960, image1280x960Modifed);
		//modifyColorsThreadsForEach(image2048x1536, image2048x1536Modifed);
		//modifyColorsThreadsForEach(image7680x4320, image7680x4320Modifed);

		//modifyColorsBlueThreadsForEach(image7680x4320, image7680x4320Modifed);
		//modifyColorsYellowThreadsForEach(image7680x4320, image7680x4320Modifed);

		end = chrono::high_resolution_clock::now(); //Точка для конца счета времени
		severalThreadDuration.emplace_back(end - start);
		// ================================================================

		// =============== Однопоточная модификация каналов ===============
		start = chrono::high_resolution_clock::now(); //Точка для начала счета времени

		//modifyColors(image1024x768, image1024x768Modifed);
		//modifyColors(image1280x960, image1280x960Modifed);
		modifyColors(image2048x1536, image2048x1536Modifed);
		//modifyColors(image7680x4320, image7680x4320Modifed); 

		//modifyColorsBlue(image7680x4320, image7680x4320Modifed);
		//modifyColorsYellow(image7680x4320, image7680x4320Modifed);

		end = chrono::high_resolution_clock::now(); //Точка для конца счета времени
		oneThreadDuration.emplace_back(end - start);
		// ================================================================
		
	}
	cout << "||" << endl << endl;

	//imshow("Result image 1024x768", image1024x768Modifed);  waitKey(0);
	//imshow("Result image 1280x960", image1280x960Modifed);  waitKey(0);
	//imshow("Result image 2048x1536", image2048x1536Modifed);  waitKey(0);
	//imshow("Result image 7680x4320", image7680x4320Modifed);  waitKey(0);

	//Вывод временных показателей для многопоточного режима
	float severalThreadDurationTime = 0; int i = 0;
	cout << "All several threads calculation times: ";
	for (auto& time : severalThreadDuration)
	{
		cout << endl << "Repeat " << i << ": " << time.count(); i++;
		severalThreadDurationTime += time.count();
	}
	cout << "\nGeneral time of several threads calculation    " << severalThreadDurationTime << endl;
	cout << "Average time of several threads calculation    " << severalThreadDurationTime / repeats << endl << endl;

	//Вывод временных показателей для однопоточного режима
	float oneThreadDurationTime = 0; i = 0;
	cout << "All one thread calculation times: ";
	for (auto& time : oneThreadDuration)
	{
		cout << endl << "Repeat " << i << ": " << time.count(); i++;
		oneThreadDurationTime += time.count();
	}
	cout << "\nGeneral time of one thread calculation: " << oneThreadDurationTime << endl;
	cout << "Average time of one thread calculation " << oneThreadDurationTime / repeats << endl << endl;

	return 0;
}

/// ============================================================================= ///
/// ============================================================================= ///
/// =============================== RESERVED CODE =============================== ///
/// ============================================================================= ///
/// ============================================================================= ///

void bgr2cmykThreads1(Mat& img, vector<Mat>& cmyk)
{
	// Allocate cmyk to store 4 componets
	for (int i = 0; i < 4; i++)
	{
		cmyk.push_back(Mat(img.size(), CV_8UC1));
	}

	// Get rgb
	vector<Mat> rgb;
	split(img, rgb);

	// rgb to cmyk
	parallel_for_(Range(0, img.rows), [&](const Range& range)
		{
			for (int i = range.start; i < range.end; i++)
			{
				for (int j = 0; j < img.cols; j++)
				{
					float r = (int)rgb[2].at<uchar>(i, j) / 255.;
					float g = (int)rgb[1].at<uchar>(i, j) / 255.;
					float b = (int)rgb[0].at<uchar>(i, j) / 255.;
					float k = min(min(1 - r, 1 - g), 1 - b);

					cmyk[0].at<uchar>(i, j) = (1 - r - k) / (1 - k) * 255.;
					cmyk[1].at<uchar>(i, j) = (1 - g - k) / (1 - k) * 255.;
					cmyk[2].at<uchar>(i, j) = (1 - b - k) / (1 - k) * 255.;
					cmyk[3].at<uchar>(i, j) = k * 255.;
				}
			}
			cout << "CMYK I am Thread: " << this_thread::get_id() << endl;
		}, 6);

}

void bgr2cmykThreads2(Mat& img, vector<Mat>& cmyk)
{
	// Allocate cmyk to store 4 componets
	for (int i = 0; i < 4; i++)
	{
		cmyk.push_back(Mat(img.size(), CV_8UC1));
	}

	// Get rgb
	vector<Mat> rgb;
	split(img, rgb);

	// rgb to cmyk

#pragma omp parallel for num_threads(6)
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			float r = (int)rgb[2].at<uchar>(i, j) / 255.;
			float g = (int)rgb[1].at<uchar>(i, j) / 255.;
			float b = (int)rgb[0].at<uchar>(i, j) / 255.;
			float k = min(min(1 - r, 1 - g), 1 - b);

			cmyk[0].at<uchar>(i, j) = (1 - r - k) / (1 - k) * 255.;
			cmyk[1].at<uchar>(i, j) = (1 - g - k) / (1 - k) * 255.;
			cmyk[2].at<uchar>(i, j) = (1 - b - k) / (1 - k) * 255.;
			cmyk[3].at<uchar>(i, j) = k * 255.;
		}
	}
}
void modifyColorsBlueThreads2(const Mat& imageBGR, Mat& imageResult)
{
	//cout << "\nFilter Blue is start";

	imageBGR.copyTo(imageResult);

	uchar B, G, R = 0;
#pragma omp parallel for num_threads(6)
	for (int i = 0; i < imageBGR.rows; i++)
	{
		for (int j = 0; j < imageBGR.cols; j++)
		{
			B = imageBGR.at<Vec3b>(i, j)[0];
			G = imageBGR.at<Vec3b>(i, j)[1];
			R = imageBGR.at<Vec3b>(i, j)[2];
			imageResult.at<Vec3b>(i, j)[0] = B - (G + B) / 2;
		}
	}
	//imshow("Modified Blue", imageResult);  waitKey(0);

	//cout << "\nFilter Blue is end";
}
void modifyColorsYellowThreads2(const Mat& imageBGR, Mat& imageResult)
{
	//cout << "\nFilter Yellow is start";

	imageBGR.copyTo(imageResult);

	vector<Mat> imageCMYK;
	bgr2cmykThreads2(imageResult, imageCMYK);

	uchar B, G, R = 0;
#pragma omp parallel for num_threads(6)
	for (int i = 0; i < imageBGR.rows; i++)
	{
		for (int j = 0; j < imageBGR.cols; j++)
		{
			if (imageCMYK[2].at<uchar>(i, j))
			{
				B = imageBGR.at<Vec3b>(i, j)[0];
				G = imageBGR.at<Vec3b>(i, j)[1];
				R = imageBGR.at<Vec3b>(i, j)[2];
				imageResult.at<Vec3b>(i, j) = R + G - 2 * (abs(R - G) + B);
			}
		}
	}
	//imshow("Modified Yellow", imageResult);  waitKey(0);

	//cout << "\nFilter Yellow is end";
}
void modifyColorsThreads2(const Mat& imageBGR, Mat& imageResult)
{
	//cout << "\nFilter is start";

	modifyColorsBlueThreads2(imageBGR, imageResult);
	modifyColorsYellowThreads2(imageResult, imageResult);

	//cout << "\nFilter is end";
}