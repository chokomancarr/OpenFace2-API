#include "LandmarkCoreIncludes.h"
#include "GazeEstimation.h"
#include <opencv2/opencv.hpp>

#include <Visualizer.h>
#include <VisualizationUtils.h>

// std::vector<std::string> get_arguments(int argc, char **argv)
// {

// 	std::vector<std::string> arguments;

// 	for (int i = 0; i < argc; ++i)
// 	{
// 		arguments.push_back(std::string(argv[i]));
// 	}
// 	return arguments;
// }

int main(int argc, char **argv) {
    // Use default parameters. 
    LandmarkDetector::FaceModelParameters parameters;

    // The modules that are being used for tracking
    // lib\local\LandmarkDetector\src\LandmarkDetectorModel.cpp
	LandmarkDetector::CLNF face_model(parameters.model_location);
	if (!face_model.loaded_successfully)
	{
		std::cerr << "ERROR: Could not load the landmark detector. " << std::endl;
		return 1;
	}

	if (!face_model.eye_model)
	{
		std::cerr << "WARNING: No eye model found. " << std::endl;
	}

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
		std::cerr << "ERROR: Cannot open a Webcam device. " << std::endl;
		return 1;
    }

    // lib\local\Utilities\src\Visualizer.cpp
	Utilities::Visualizer visualizer(true, false, false, false);

	int cam_width = 640;
	int cam_height = 480;
    float cx = cam_width / 2.0f;
    float cy = cam_height / 2.0f;
    float fx = 500.0f * (cam_width / 640.0f);
    float fy = 500.0f * (cam_height / 480.0f);

    int key = -1;
    cv::Mat frame_bgr, frame_gray;
    // while ((key = cv::waitKey(10)) != 27) {
    while (true) {
		if(!(cap.grab() && cap.retrieve(frame_bgr))){
            std::cerr << "ERROR: Failed to glab a frame. " << std::endl;
			break;
		}
        // cv::imshow("Preview", frame_bgr);
        cv::cvtColor(frame_bgr, frame_gray, cv::COLOR_BGR2GRAY);
        bool detection_success = LandmarkDetector::DetectLandmarksInVideo(frame_bgr, face_model, parameters, frame_gray);

        cv::Point3f gaze_direction_L(0, 0, -1);
        cv::Point3f gaze_direction_R(0, 0, -1);

        // If tracking succeeded and we have an eye model, estimate gaze
        if (detection_success && face_model.eye_model)
        {
            GazeAnalysis::EstimateGaze(face_model, gaze_direction_L, fx, fy, cx, cy, true);
            GazeAnalysis::EstimateGaze(face_model, gaze_direction_R, fx, fy, cx, cy, false);
        }

		cv::Vec6d pose_estimate = LandmarkDetector::GetPose(face_model, fx, fy, cx, cy);
        std::cout << "L: (" 
                << gaze_direction_L.x << ", "
                << gaze_direction_L.y << ", "
                << gaze_direction_L.z << "). " << std::endl;
        std::cout << "R: (" 
                << gaze_direction_R.x << ", "
                << gaze_direction_R.y << ", "
                << gaze_direction_R.z << ")" << std::endl;
        std::cout << "Face: (" 
                << pose_estimate[0] << ", "
                << pose_estimate[1] << ", "
                << pose_estimate[2] << ") - ("
                << pose_estimate[3] << ", "
                << pose_estimate[4] << ", "
                << pose_estimate[5] << ")" << std::endl;

        std::cout << "Landmarks: [" << std::endl;
        int num_of_landmarks = face_model.detected_landmarks.rows / 2;
        for (int i = 0; i < num_of_landmarks; i++) {
            std::cout << "\t[" << i << "]: ("
                << face_model.detected_landmarks.at<float>(i) << ", "
                << face_model.detected_landmarks.at<float>(i+num_of_landmarks) << ")" << std::endl;
        }
        std::cout << "]" << std::endl;


        visualizer.SetImage(frame_bgr, fx, fy, cx, cy);
        visualizer.SetObservationLandmarks(face_model.detected_landmarks, face_model.detection_certainty, face_model.GetVisibilities());
        visualizer.SetObservationPose(pose_estimate, face_model.detection_certainty);
        visualizer.SetObservationGaze(
            gaze_direction_L, gaze_direction_R, 
            LandmarkDetector::CalculateAllEyeLandmarks(face_model), 
            LandmarkDetector::Calculate3DEyeLandmarks(face_model, fx, fy, cx, cy), 
            face_model.detection_certainty);

        if(visualizer.ShowObservation() == 27) { break; }
    }

}